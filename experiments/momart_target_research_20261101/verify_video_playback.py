"""Decode a real MP4 in ffmpeg and Chrome, checking visible canvas pixels."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
import tempfile
import imageio_ffmpeg


def main(a):
    video=a.video.resolve()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg,'-v','error','-i',str(video),'-f','null','-'],check=True,capture_output=True)
    records=[]
    for number,second in enumerate(a.times):
        page=a.output/f'probe_{number}.html';dump=a.output/f'browser_{number}.html'
        # Each actual video/canvas is local; no authenticated browser profile or
        # account cookies are used or inspected.
        page.write_text('''<!doctype html><meta charset="utf-8"><video id="v" muted preload="auto" src="'''+html.escape(video.as_uri(),quote=True)+'''"></video>
<canvas id="c" width="320" height="180"></canvas><pre id="result">PENDING</pre><script>
const v=document.getElementById('v'),c=document.getElementById('c'),r=document.getElementById('result');
v.addEventListener('error',()=>r.textContent=JSON.stringify({ok:false,error:v.error?.code,message:v.error?.message}));
v.addEventListener('loadedmetadata',()=>v.currentTime='''+json.dumps(second)+''');
v.addEventListener('seeked',async()=>{try{await v.play();const x=c.getContext('2d');x.drawImage(v,0,0,320,180);
const p=x.getImageData(0,0,320,180).data;let sum=0,sum2=0;
for(let i=0;i<p.length;i+=4){const n=(p[i]+p[i+1]+p[i+2])/3;sum+=n;sum2+=n*n;}const mean=sum/(p.length/4);
r.textContent=JSON.stringify({ok:true,width:v.videoWidth,height:v.videoHeight,time:v.currentTime,duration:v.duration,playing:!v.paused,mean,std:Math.sqrt(sum2/(p.length/4)-mean*mean)});
}catch(e){r.textContent=JSON.stringify({ok:false,error:String(e)});}});</script>''')
        with tempfile.TemporaryDirectory(prefix='momart_video_browser_') as profile:
            result=subprocess.run([a.browser,'--headless','--no-sandbox','--disable-gpu','--allow-file-access-from-files',
                '--autoplay-policy=no-user-gesture-required','--user-data-dir='+profile,'--virtual-time-budget=10000','--dump-dom',page.resolve().as_uri()],
                capture_output=True,text=True,timeout=30,check=True)
        dump.write_text(result.stdout)
        match=re.search(r'<pre id="result">(.*?)</pre>',result.stdout,re.S)
        if match is None or match.group(1)=='PENDING':raise ValueError('Browser video did not render')
        value=json.loads(html.unescape(match.group(1)))
        if not (value.get('ok') and value.get('playing') and value.get('mean',0)>3 and value.get('std',0)>3):
            raise ValueError('Browser playback failed or rendered black: '+str(value))
        value['requested_time']=second;records.append(value)
        subprocess.run([ffmpeg,'-y','-v','error','-ss',str(second),'-i',str(video),'-frames:v','1',str(a.output/f'frame_{number}.png')],check=True,capture_output=True)
    report=dict(video=str(video),sha256=hashlib.sha256(video.read_bytes()).hexdigest(),full_ffmpeg_decode=True,browser=a.browser,
        browser_play_and_canvas=records,scope='Browser decode of supplied file, including cloud-transcoded previews when downloaded; not an authenticated Feishu UI or user-device guarantee')
    (a.output/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--times',type=float,nargs='+',default=[.1,2.]);p.add_argument('--browser',default='/usr/bin/google-chrome');main(p.parse_args())
