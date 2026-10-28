#!/usr/bin/env python3
"""Editable, evidence-grounded MoMaRT method figures; no model execution.

Run with a Python environment containing cairosvg to also emit PNG and PDF.
Both SVGs share geometry and stable module IDs for visual revision comparison.
"""
from __future__ import annotations

import html
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs/assets/momart_main_figures_20261028/native"
ARTIFACTS = ROOT / "artifacts/momart_main_figures_20261028"
W, H = 2000, 1200
INK = "#203342"
MUTED = "#667986"
BLUE = "#5088B3"
BLUE_PALE = "#E7F1F8"
GREEN = "#6E9C88"
GREEN_PALE = "#ECF4EF"
PINK = "#B96F81"
PINK_PALE = "#F9EDF0"
RED = "#D64D58"
GRAY = "#9AA6AD"
LIGHT = "#D6E0E5"


class Canvas:
    def __init__(self, boxed: bool):
        self.boxed = boxed
        self.p = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title description">
<title id="title">{'Boxed ' if boxed else ''}No-History: MoMaRT policy architecture</title>
<desc id="description">Two recent multimodal observations are encoded into two 256-dimensional tokens, with a task embedding. A PTP Transformer predicts action noise over a nine-step, ten-dimensional action chunk. Long-term episodic memory is removed. {'Only main RGB receives a red box cue; other modalities are unchanged.' if boxed else 'No long-history cache or compressor is used.'}</desc>
<defs>
 <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10Z" fill="{INK}"/></marker>
 <marker id="arrow-blue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10Z" fill="{BLUE}"/></marker>
 <marker id="arrow-gray" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10Z" fill="{GRAY}"/></marker>
 <style>text{{font-family:'DejaVu Sans','Arial',sans-serif;fill:{INK};}} .label{{font-size:22px}} .small{{font-size:18px}}</style>
</defs>
<rect width="2000" height="1200" fill="white"/>
''']

    def add(self, value):
        self.p.append(value)

    def group(self, name):
        self.add(f'<g id="{name}">')

    def end(self):
        self.add('</g>')

    def rect(self, x, y, w, h, fill="white", stroke=LIGHT, rx=12, sw=1.7, dash=None):
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>')

    def text(self, x, y, value, size=22, weight=400, color=INK, anchor="start"):
        self.add(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}" style="fill:{color}">{html.escape(value)}</text>')

    def path(self, d, color=INK, sw=2.1, arrow=True, dash=None):
        marker = "arrow-blue" if color == BLUE else "arrow-gray" if color == GRAY else "arrow"
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"'+(f' marker-end="url(#{marker})"' if arrow else '')+(f' stroke-dasharray="{dash}"' if dash else '')+'/>')

    def line(self, x1, y1, x2, y2, **kw):
        self.path(f'M{x1} {y1}L{x2} {y2}', **kw)

    def circle(self, x, y, r, fill="white", stroke=INK, sw=2):
        self.add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def panel(self, name, x, y, w, h, label, title):
        self.group(name)
        self.rect(x, y, w, h, stroke=LIGHT, rx=17)
        self.text(x+24,y+43,label,24,700,color=BLUE)
        self.text(x+71,y+43,title,25,650)
        self.end()

    def token(self, x, y, w=46, h=68, color=BLUE_PALE, stroke=BLUE, label=None):
        self.rect(x,y,w,h,color,stroke,rx=5,sw=1.5)
        for k in range(4):
            self.line(x+9,y+14+k*11,x+w-9,y+14+k*11,color=stroke,sw=1,arrow=False)
        if label:
            self.text(x+w/2,y+h+25,label,18,anchor="middle")

    def image_card(self, x, y, label, depth=False, boxed=False, wrist=False):
        # Explicitly schematic observations; never presented as actual rollout frames.
        self.rect(x-7,y-7,171,106,"white",BLUE,rx=6,sw=1)
        self.rect(x,y,171,106,"#E4E9EB" if depth else "#EAE9E4",BLUE,rx=6,sw=1.6)
        self.path(f'M{x+2} {y+67}L{x+169} {y+67}',color="#BAC6C9",arrow=False,sw=2)
        self.add(f'<path d="M{x+2} {y+67}L{x+48} {y+47}L{x+169} {y+47}L{x+169} {y+85}L{x+2} {y+85}Z" fill="{"#A6B1B7" if depth else "#C9C5B9"}"/>')
        self.rect(x+20,y+12,45,29,"#B2BCC2" if depth else "#D0D8DA","#ABB7BD",rx=1,sw=1)
        self.line(x+43,y+13,x+43,y+40,color="#8C9CA6",arrow=False,sw=1)
        # Red target bowl on counter; geometry is a schematic cue.
        target = "#7E8D98" if depth else "#B85C62"
        self.add(f'<path d="M{x+93} {y+57}L{x+126} {y+57}Q{x+122} {y+73} {x+110} {y+73}Q{x+97} {y+73} {x+93} {y+57}Z" fill="{target}"/>')
        self.add(f'<ellipse cx="{x+109.5}" cy="{y+57}" rx="16.5" ry="4" fill="{"#9BA7B0" if depth else "#D28488"}"/>')
        if wrist:
            self.path(f'M{x+30} {y+102}L{x+58} {y+76}L{x+84} {y+69}',color="#7D909C",sw=9,arrow=False)
            self.path(f'M{x+83} {y+62}L{x+92} {y+62}M{x+84} {y+77}L{x+93} {y+77}',color="#657A87",sw=3,arrow=False)
        else:
            self.path(f'M{x+27} {y+102}L{x+37} {y+78}L{x+65} {y+68}',color="#83929A",sw=8,arrow=False)
            self.circle(x+37,y+78,5,"#B9C4C9","#71858F",1)
        if boxed:
            self.rect(x+88,y+49,44,29,"none",RED,rx=0,sw=2)
        self.text(x+85.5,y+132,label,20,550,anchor="middle")


def draw(boxed: bool) -> str:
    c=Canvas(boxed)
    c.group("figure-header")
    c.text(52,57,"MoMaRT  /  METHOD OVERVIEW",18,650,color=MUTED)
    c.text(52,112,"Boxed No-History" if boxed else "No-History",43,700)
    c.text(535,108,"Target cues in pixels, recent observations for control" if boxed else "A short-context policy derived from VPWEM",25,color=MUTED)
    c.text(1945,56,"A · EDITABLE VECTOR",16,650,color=MUTED,anchor="end")
    c.line(52,140,1948,140,color=LIGHT,sw=1.2,arrow=False)
    c.end()

    c.panel("observation-panel",50,174,470,658,"a","Recent observations")
    c.panel("encoding-panel",565,174,395,658,"b","Multimodal encoding")
    c.panel("policy-panel",1010,174,940,658,"c","Conditional action diffusion")

    c.group("recent-observations")
    c.rect(76,239,418,40,BLUE_PALE,BLUE,rx=8,sw=1)
    c.text(285,265,"Two steps:  oₜ₋₁, oₜ",21,550,anchor="middle")
    c.image_card(87,314,"Main RGB",boxed=boxed)
    c.image_card(305,314,"Wrist RGB",wrist=True)
    c.image_card(87,491,"Main depth",depth=True)
    c.image_card(305,491,"Wrist depth",depth=True,wrist=True)
    c.text(285,658,"90 × 120 pixels per view · schematic",18,color=MUTED,anchor="middle")
    c.rect(77,688,198,77,"#F6F8F9",LIGHT,rx=8)
    c.path('M96 730L119 711L140 735L164 715L184 733',color=BLUE,sw=3,arrow=False)
    for x,y in [(96,730),(119,711),(140,735),(164,715),(184,733)]: c.circle(x,y,3,BLUE,BLUE,1)
    c.text(176,755,"Proprio · 10",18,anchor="middle")
    c.rect(290,688,202,77,"#F6F8F9",LIGHT,rx=8)
    c.path('M309 732L322 722L333 730L347 710L357 722L373 711L389 731L404 713L423 725L444 717L471 727',color=BLUE,sw=2,arrow=False)
    c.text(391,755,"Scan · 1 × 100",18,anchor="middle")
    if boxed:
        c.text(285,802,"Red outline only on main RGB",20,600,color=RED,anchor="middle")
    else:
        c.text(285,802,"All observation modalities retained",19,550,color=MUTED,anchor="middle")
    c.end()

    c.group("multimodal-encoder")
    c.text(762,264,"Per step; shared across time",18,550,color=MUTED,anchor="middle")
    c.rect(591,296,343,262,GREEN_PALE,GREEN,rx=12)
    c.text(762,328,"4 independent ResNet-18s",20,650,anchor="middle")
    for i,label in enumerate(["RGB","Wrist RGB","Depth","Wrist depth"]):
        y=347+i*47
        c.rect(609,y,197,35,"white",GREEN,rx=5,sw=1)
        c.text(707,y+24,label,18,anchor="middle")
        c.line(813,y+17,845,y+17,color=GREEN,sw=1.5)
        c.text(871,y+25,"512",18,550,anchor="middle")
    c.text(762,543,"Crop 75 × 100 · GroupNorm",17,color=MUTED,anchor="middle")
    c.line(521,425,562,425,color=BLUE)
    c.line(521,726,550,726,color=BLUE,arrow=False)
    c.path('M550 726L550 613L590 613',color=BLUE)
    c.rect(591,584,343,66,GREEN_PALE,GREEN,rx=8)
    c.line(762,592,762,642,color=GREEN,sw=1,arrow=False)
    c.text(672,611,"Scan encoder",18,600,anchor="middle")
    c.text(672,635,"1D conv → 64",17,anchor="middle")
    c.text(852,611,"Proprio",18,600,anchor="middle")
    c.text(852,635,"10 → identity",17,anchor="middle")
    c.path('M934 528L946 528L946 670L762 670L762 681',color=GREEN)
    c.path('M672 650L672 670L762 670',color=GREEN,arrow=False)
    c.path('M852 650L852 670L762 670',color=GREEN,arrow=False)
    c.rect(591,684,343,103,GREEN_PALE,GREEN,rx=10)
    c.text(762,715,"Concatenate → MLP",22,650,anchor="middle")
    c.text(762,745,"2122 → 256 → 256",23,600,anchor="middle")
    c.text(762,773,"4 × 512 + 64 + 10 = 2122",17,color=MUTED,anchor="middle")
    c.path('M934 736L985 736L985 347L1053 347',color=BLUE,sw=2.5)
    c.end()

    c.group("working-memory")
    c.text(1108,269,"Obs. tokens",20,600,anchor="middle")
    c.token(1055,305,label="oₜ₋₁")
    c.token(1118,305,label="oₜ")
    c.text(1108,437,"2 × 256",20,600,anchor="middle")
    c.circle(1220,340,19,stroke=BLUE)
    c.text(1220,349,"+",28,500,color=BLUE,anchor="middle")
    c.line(1171,340,1197,340,color=BLUE)
    c.rect(1181,242,236,52,GREEN_PALE,GREEN,rx=8)
    c.text(1299,275,"Task embedding",21,600,anchor="middle")
    c.path('M1299 294L1299 307L1220 307L1220 318',color=GREEN)
    c.line(1243,340,1292,340,color=BLUE)
    c.text(1350,379,"condition",17,color=MUTED,anchor="middle")
    c.token(1301,310,w=34,h=61)
    c.token(1344,310,w=34,h=61)
    c.line(1388,340,1462,340,color=BLUE)
    c.end()

    c.group("condition-sequence")
    c.rect(1474,240,443,226,"#F6F9FB",BLUE,rx=11,sw=1.4)
    c.text(1695,273,"PTP condition sequence",22,650,anchor="middle")
    c.token(1528,310,w=48,h=65,color=PINK_PALE,stroke=PINK,label="τ")
    c.token(1603,310,w=48,h=65,label="oₜ₋₁")
    c.token(1678,310,w=48,h=65,label="oₜ")
    c.text(1834,336,"Tₘ = 0",25,650,anchor="middle")
    c.text(1834,367,"no episodic",17,color=MUTED,anchor="middle")
    c.text(1834,389,"tokens",17,color=MUTED,anchor="middle")
    c.text(1695,443,"[B, 3, 256] = time + two recent tokens",18,color=MUTED,anchor="middle")
    c.path('M1695 466L1695 507L1459 507L1459 548',color=BLUE,sw=2.5)
    c.text(1469,498,"cross-attention",18,color=BLUE)
    c.end()

    c.group("action-denoiser")
    c.text(1130,540,"Noisy actions  Aᵏ",20,600,anchor="middle")
    for i in range(9): c.rect(1048+i*18,579,14,66,"#DFE3E6" if i%2 else "#EFF1F3",GRAY,rx=2,sw=1)
    c.text(1130,676,"[B, 9, 10]",19,600,anchor="middle")
    c.line(1220,612,1280,612)
    c.rect(1297,544,352,158,"white",GREEN,rx=11,sw=1)
    c.rect(1289,552,352,158,"white",GREEN,rx=11,sw=1)
    c.rect(1281,560,352,158,GREEN_PALE,GREEN,rx=11,sw=2)
    c.text(1457,598,"PTP Transformer",27,650,anchor="middle")
    c.text(1457,630,"8 layers · 4 heads · d = 256",19,anchor="middle")
    c.text(1457,662,"causal attention + aligned context",17,color=MUTED,anchor="middle")
    c.text(1457,693,"Linear → decoder → noise head",17,color=MUTED,anchor="middle")
    c.line(1644,612,1706,612)
    c.text(1803,540,"Predicted noise  εθ",20,600,anchor="middle")
    for i in range(9): c.rect(1721+i*18,579,14,66,PINK_PALE,PINK,rx=2,sw=1)
    c.text(1803,676,"[B, 9, 10]",19,600,anchor="middle")
    c.path('M1810 697L1810 772L1129 772L1129 697',color=PINK,sw=2)
    c.rect(1295,752,324,39,"white","white",rx=0,sw=0)
    c.text(1457,778,"50 DDPM denoising steps",20,600,color=PINK,anchor="middle")
    c.end()

    c.group("method-change-panel")
    c.rect(50,869,910,262,"#FCFCFD",LIGHT,rx=15)
    if boxed:
        c.text(76,910,"d",24,700,color=RED)
        c.text(123,910,"Target-box cue generation",26,650)
        c.rect(76,937,268,63,PINK_PALE,PINK,rx=8)
        c.text(210,963,"Training",19,650,anchor="middle")
        c.text(210,987,"offline Grounding DINO",18,anchor="middle")
        c.rect(76,1011,268,63,PINK_PALE,PINK,rx=8)
        c.text(210,1037,"Deployment",19,650,anchor="middle")
        c.text(210,1061,"online combined_blend",18,anchor="middle")
        c.path('M344 968L380 968L380 1005L410 1005',color=PINK)
        c.path('M344 1042L380 1042L380 1005',color=PINK,arrow=False)
        c.rect(421,958,236,93,"white",RED,rx=8)
        c.text(539,990,"Draw red outline",21,650,anchor="middle")
        c.text(539,1020,"main RGB · 1 pixel",18,anchor="middle")
        c.line(658,1005,704,1005,color=BLUE)
        c.text(813,986,"Recent RGB",23,650,anchor="middle")
        c.text(813,1015,"observation",21,anchor="middle")
        c.text(813,1043,"→ panel a",19,color=BLUE,anchor="middle")
        c.text(76,1103,"Missing box → original RGB. No ROI tokens; other modalities unchanged.",18,color=MUTED)
    else:
        c.text(76,910,"d",24,700,color=GRAY)
        c.text(123,910,"Removed from VPWEM: episodic-memory branch",25,650)
        c.rect(76,947,214,66,"white",GRAY,rx=7,dash="6 5")
        c.text(183,975,"Observation cache",19,550,color=MUTED,anchor="middle")
        c.text(183,999,"out-of-window frames",16,color=MUTED,anchor="middle")
        c.rect(76,1028,214,59,"white",GRAY,rx=7,dash="6 5")
        c.text(183,1064,"Summary cache",19,550,color=MUTED,anchor="middle")
        c.path('M290 980L333 980L333 1019L372 1019',color=GRAY,dash="5 5")
        c.path('M290 1057L333 1057L333 1019',color=GRAY,dash="5 5",arrow=False)
        c.rect(382,955,265,130,"white",GRAY,rx=9,dash="6 5")
        c.text(514,985,"Q-Former compressor",20,600,color=MUTED,anchor="middle")
        c.text(514,1014,"summary self-attention",17,color=MUTED,anchor="middle")
        c.text(514,1041,"observation cross-attention",17,color=MUTED,anchor="middle")
        c.text(514,1068,"learned queries + FFN",17,color=MUTED,anchor="middle")
        c.line(655,1019,707,1019,color=GRAY,dash="5 5")
        c.token(728,982,color="#F2F3F4",stroke=GRAY)
        c.token(793,982,color="#F2F3F4",stroke=GRAY)
        c.line(716,1070,853,969,color=PINK,sw=3,arrow=False)
        c.text(784,1099,"No long-history tokens",18,600,color=MUTED,anchor="middle")
    c.end()

    c.group("training-and-execution")
    c.rect(1010,869,940,262,"white",LIGHT,rx=15)
    c.text(1035,910,"e",24,700,color=BLUE)
    c.text(1082,910,"Training and closed-loop execution",25,650)
    c.text(1038,953,"Train",19,650,color=GREEN)
    c.text(1116,953,"clean actions + Gaussian noise → predict ε → denoising loss",19)
    for x,w,title,sub in [(1037,233,"Observe","latest 2 steps"),(1325,267,"Denoise","9 × 10 action chunk"),(1650,272,"Execute","8 actions, then replan")]:
        c.rect(x,979,w,85,BLUE_PALE if title!="Denoise" else GREEN_PALE,BLUE if title!="Denoise" else GREEN,rx=9)
        c.text(x+w/2,1012,title,23,650,anchor="middle")
        c.text(x+w/2,1042,sub,18,anchor="middle")
    c.line(1276,1021,1317,1021,color=BLUE)
    c.line(1598,1021,1642,1021,color=BLUE)
    c.path('M1786 1066L1786 1096L1154 1096L1154 1066',color=BLUE,sw=1.8)
    c.rect(1335,1081,259,31,"white","white",rx=0,sw=0)
    c.text(1465,1103,"no policy history cache",18,color=MUTED,anchor="middle")
    c.end()

    c.group("figure-footer")
    c.text(53,1170,"Blue: recent observations / control     Green: learned policy modules     Pink: diffusion / cue operations",17,color=MUTED)
    c.text(1947,1170,"Architecture only · scores reported separately",17,color=MUTED,anchor="end")
    c.end()
    c.add('</svg>')
    return ''.join(c.p)


def main():
    SOURCE.mkdir(parents=True,exist_ok=True)
    ARTIFACTS.mkdir(parents=True,exist_ok=True)
    import cairosvg
    for name,boxed in [("no_history",False),("boxed",True)]:
        svg=SOURCE/f"{name}.svg"
        svg.write_text(draw(boxed),encoding="utf-8")
        cairosvg.svg2png(url=str(svg),write_to=str(ARTIFACTS/f"native_{name}.png"),output_width=3000,output_height=1800)
        cairosvg.svg2pdf(url=str(svg),write_to=str(ARTIFACTS/f"native_{name}.pdf"))
        print(svg.relative_to(ROOT))


if __name__ == "__main__":
    main()
