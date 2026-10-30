"""Development probe only: fixed audit frames, no cache or training mutations.

Compare six independent prompts (with/without returned-phrase verification) to
one joint caption scored by explicit phrase token spans, as in the official
GroundingDINO demo. Mean-token scores differ from max-token scores; equal
thresholds are exploratory, not calibrated operating points.
"""
import argparse
import inspect
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from momart_bbox_memory.detector import MODEL_ID, REVISION, red_support_filter
from momart_object_memory.perception import bounded_detections
from momart_object_memory.state import iou

PHRASES = ['a red bowl', 'a dishwasher door', 'a dishwasher door handle',
           'a dishwasher rack', 'a drawer', 'a drawer handle']
NAMES = ['bowl', 'door', 'handle', 'rack', 'drawer', 'drawer_handle']


def phrase_matches(label, phrase):
    """Require all and only requested content words, ignoring articles/punctuation."""
    def words(value):
        return [w for w in re.findall(r'[a-z]+', value.lower()) if w not in ('a', 'an', 'the')]
    return words(label) == words(phrase)


def select(image, candidates):
    kept = []
    for cls in range(len(PHRASES)):
        subset = sorted([d for d in candidates if d[0] == cls], key=lambda d: -d[2])
        selected = []
        for item in subset:
            box = item[1]
            if not np.isfinite(box).all() or not np.isfinite(item[2]):
                continue
            if np.any(box[2:] <= box[:2]):
                continue
            if cls == 0 and not red_support_filter(image, [box * [120, 90, 120, 90]])[0]:
                continue
            if all(iou(box, other[1]) <= .5 for other in selected):
                selected.append(item)
        kept.extend(selected)
    return [dict(class_id=c, box=b.tolist(), score=s, label=label)
            for c, b, s, label in bounded_detections(kept, 8)]


def main():
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True)
    processor.image_processor.size = dict(shortest_edge=600, longest_edge=1333)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        MODEL_ID, revision=REVISION, local_files_only=True).cuda().eval().requires_grad_(False)
    sample = json.loads((args.input / 'sample.json').read_text())
    caption = '. '.join(PHRASES) + '.'
    offsets = processor.tokenizer(caption, return_offsets_mapping=True)['offset_mapping']
    spans = []
    start = 0
    for phrase in PHRASES:
        start = caption.index(phrase, start)
        spans.append([i for i, (lo, hi) in enumerate(offsets)
                      if hi > lo and lo < start + len(phrase) and hi > start])
        start += len(phrase)
    records = []
    with torch.inference_mode():
        for sample_item in sample['samples']:
            frame = sample_item['frame']
            pil = Image.open(args.input / ('frame_%s.png' % frame)).convert('RGB')
            image = np.asarray(pil)
            torch.cuda.synchronize()
            begin = time.perf_counter()
            inputs = processor(images=[pil] * 6, text=[p + '.' for p in PHRASES],
                               padding=True, return_tensors='pt').to('cuda')
            outputs = model(**inputs)
            post = processor.post_process_grounded_object_detection
            key = 'box_threshold' if 'box_threshold' in inspect.signature(post).parameters else 'threshold'
            raw = post(outputs, inputs.input_ids, **{key: .25}, text_threshold=.25,
                       target_sizes=[(90, 120)] * 6)
            candidates = []
            for cls, result in enumerate(raw):
                labels = result.get('text_labels', result.get('labels'))
                for box, score, label in zip(result['boxes'], result['scores'], labels):
                    candidates.append((cls, np.clip(box.cpu().numpy()/[120,90,120,90],0,1).astype(np.float32),
                                       float(score), label))
            six = select(image, candidates)
            strict = select(image, [x for x in candidates if phrase_matches(x[3], PHRASES[x[0]])])
            torch.cuda.synchronize()
            six_seconds = time.perf_counter() - begin
            begin = time.perf_counter()
            inputs = processor(images=pil, text=caption, return_tensors='pt').to('cuda')
            outputs = model(**inputs)
            probabilities = outputs.logits[0].sigmoid()
            raw_boxes = outputs.pred_boxes[0]
            boxes = torch.cat((raw_boxes[:, :2] - raw_boxes[:, 2:]/2,
                               raw_boxes[:, :2] + raw_boxes[:, 2:]/2), dim=-1).clamp(0,1)
            joint = []
            for cls, token_ids in enumerate(spans):
                scores = probabilities[:, token_ids].mean(-1)
                for idx in torch.where(scores > .25)[0].tolist():
                    joint.append((cls, boxes[idx].cpu().numpy(), float(scores[idx]), PHRASES[cls]))
            joint = select(image, joint)
            torch.cuda.synchronize()
            joint_seconds = time.perf_counter() - begin
            records.append(dict(frame=frame, six=six, strict=strict, joint=joint,
                                six_seconds=six_seconds, joint_seconds=joint_seconds))
            print(json.dumps(dict(frame=frame, six=len(six), strict=len(strict), joint=len(joint))), flush=True)
    result = dict(model=MODEL_ID, revision=REVISION, caption=caption, token_spans=spans,
                  threshold=.25, text_threshold=.25, nms=.5, max_detections=8,
                  note='Development frames from one dataset. No GT, no calibrated precision/recall.',
                  records=records)
    (args.output/'probe.json').write_text(json.dumps(result, indent=2))
    for cls in (2, 5):
        sheet = Image.new('RGB', (900, 36*120), 'white')
        draw = ImageDraw.Draw(sheet)
        for row, record in enumerate(records):
            original = Image.open(args.input/('frame_%s.png'%record['frame'])).convert('RGB').resize((144,108))
            sheet.paste(original, (0,row*120))
            for col, mode in enumerate(('six','strict','joint')):
                picture = original.copy()
                pen = ImageDraw.Draw(picture)
                found = [x for x in record[mode] if x['class_id']==cls]
                for x in found:
                    b = np.array(x['box'])*[144,108,144,108]
                    pen.rectangle(tuple(b), outline='red', width=2)
                sheet.paste(picture, (180+col*230,row*120))
                draw.text((180+col*230,row*120+108), '%s f%s n%s'%(mode,record['frame'],len(found)),fill='black')
        sheet.save(args.output/('class_%s.png'%cls))


if __name__ == '__main__':
    main()
