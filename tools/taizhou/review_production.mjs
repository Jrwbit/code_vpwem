// Actual deployed HTML/scripts and GET feedback; every POST is fenced locally.
import fs from 'node:fs';
const cdp=process.env.CDP_URL||'http://127.0.0.1:19337';
const origin=process.env.R435_REVIEW_URL||'http://127.0.0.1:18768/';
const tab=await(await fetch(cdp+'/json/new?about:blank',{method:'PUT'})).json();
const ws=new WebSocket(tab.webSocketDebuggerUrl);
await new Promise(r=>ws.addEventListener('open',r,{once:true}));
let seq=0;const pending=new Map(),errors=[];
ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails);};
const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});
const ev=async expression=>{const r=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true,userGesture:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
const pause=ms=>new Promise(r=>setTimeout(r,ms));
const assert=(v,msg)=>{if(!v)throw Error(msg);};
const evidence=[];
try{
 await call('Runtime.enable');await call('Page.enable');
 await call('Emulation.setFocusEmulationEnabled',{enabled:true});await call('Page.bringToFront');
 await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
 await call('Page.addScriptToEvaluateOnNewDocument',{source:`
 const realFetch=window.fetch;window.__productionPosts=[];let sandboxConfig;window.__fixturePower=null;
 window.fetch=async(url,options={})=>{
  const path=new URL(url,location.href).pathname;
  const method=options.method||'GET';
  if(method==='POST'){
   const body=JSON.parse(options.body||'{}');window.__productionPosts.push({path,body:{...body,motion_token:body.motion_token?'present':undefined}});
   const json=(data,status=200)=>Promise.resolve(new Response(JSON.stringify(data),{status,headers:{'Content-Type':'application/json'}}));
   if(path==='/api/robot/operator/heartbeat'&&window.__fixturePower)return json({ok:true});
   if(path==='/api/robot/operator'&&window.__allowFixturePower&&body.action.endsWith('_power_on')){window.__fixturePower={prefix:body.action.startsWith('waist')?'waist':'arm',id:'fixture-owned-power'};return json({ok:true,operation_id:'fixture-owned-power'});}
   if(path==='/api/robot/operator'&&body.action.endsWith('_power_off')){window.__fixturePower=null;return json({ok:true});}
   if(path==='/api/robot/config'){sandboxConfig.config=body.config;sandboxConfig.revision+='x';return json(sandboxConfig);}
   if(path==='/api/robot/invalidate')return json({invalidated:true});
   if(path==='/api/robot/initial/save'){sandboxConfig.config.initial_T_base_flange=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];return json(sandboxConfig);}
   if(path==='/api/robot/waist/preview')return json({needs_motion:true,preview_id:'fixture-only'});
   if(path==='/api/robot/preview')return json({plan:{id:'fixture-plan',state:'ready',next_stage:0,required_confirmation:'go_initial',stages:[{name:'go_initial',label:'到达初始位置'}]},motion_enabled:true,robot:{connected:true},operator:{},power_workflow:{preconditions_met:true}});
   return json({error:'隔离按钮测试：请求已收到，未转发真机'},409);
  }
  const response=await realFetch(url,options);
  if(path==='/api/robot/status'&&window.__fixturePower){const d=await response.clone().json(),p=window.__fixturePower;d.robot.busy=true;d.robot.components[p.prefix==='waist'?'body':'left_arm'].power='on';d.operator.owned_operation_id=p.id;d.operator.operation_kind=p.prefix+'_power';d.operator.phase=p.prefix==='waist'?'waist_holding':'power_holding';d.power_workflow.preconditions_met=false;return new Response(JSON.stringify(d),{status:200,headers:{'Content-Type':'application/json'}});}
  if(path==='/api/robot/config'){
   sandboxConfig=await response.clone().json();
   sandboxConfig.config.tcp_z_adjustment_mm=0;sandboxConfig.config.fixture_press_mm={};
   sandboxConfig.config.workspace_m={x:[-1,1],y:[-1,1],z:[-1,1]};
   return new Response(JSON.stringify(sandboxConfig),{status:response.status,headers:{'Content-Type':'application/json'}});
  }
  return response;
 };`});
 await call('Page.navigate',{url:origin});await pause(6500);
 const inventory=await ev(`({checkboxes:document.querySelectorAll('#rokaeWorkbench input[type=checkbox]').length,ids:[...document.querySelectorAll('[id]')].map(e=>e.id),stages:[...document.querySelectorAll('#compactStages li')].map(e=>e.textContent),engineeringOpen:document.querySelector('.production-engineering').open,tokenPresent:!!document.getElementById('rokaeToken').value})`);
 assert(inventory.checkboxes===0,'Confirmation checkbox still exists');
 assert(new Set(inventory.ids).size===inventory.ids.length,'Duplicate DOM IDs');
 assert(inventory.stages.length===6&&!inventory.engineeringOpen&&inventory.tokenPresent,'Production layout/session failed');
 evidence.push({check:'deployed layout and session',...inventory,ids:undefined});
 const click=async(id)=>{assert(await ev(`!document.getElementById('${id}').disabled`),'Unexpected disabled '+id);await ev(`document.getElementById('${id}').click()`);await pause(500);};
 for(const [input,button,value,key] of [['compactTcpZ','compactTcpSave','6','tcp_z_adjustment_mm'],['compactPress','compactPressSave','2','fixture_press_mm'],['compactFixtureYaw','compactFixtureSave','45','tcp_T_objects'],['compactSpeed','compactSpeedSave','3','linear_speed_mm_s']]){
  await ev(`{const e=document.getElementById('${input}');e.value='${value}';e.dispatchEvent(new Event('input'));}`);await click(button);
  const saved=await ev(`window.__productionPosts.filter(p=>p.path==='/api/robot/config').at(-1).body.config`);
  assert(saved[key]!=null,'Saved parameter missing '+key);assert(saved.tcp_z_adjustment_mm===6,'Earlier saved value lost');
  evidence.push({button,check:'actual script dispatched fenced persistent config',key,value:saved[key]});
 }
 await click('compactLimitsSave');await click('compactInitialSave');
 for(const id of ['rokaeWaistClear','rokaeArmClear','rokaeWaistPower','rokaeArmPower','rokaeWaistStart']){
  await click(id);const request=await ev(`window.__productionPosts.filter(p=>p.path==='/api/robot/operator'&&p.body.action!=='stop').at(-1)`);
  const action={rokaeWaistClear:'waist_clear_alarm',rokaeArmClear:'arm_clear_alarm',rokaeWaistPower:'waist_power_on',rokaeArmPower:'arm_power_on',rokaeWaistStart:'waist_start'}[id];assert(request.body.action===action,'Wrong button action '+id);
  evidence.push({button:id,check:'actual handler; robot POST fenced',request});
 }
 await ev('window.__allowFixturePower=true');
 for(const [id,prefix] of [['rokaeWaistPower','waist'],['rokaeArmPower','arm']]){
  await click(id);await pause(800);assert(await ev(`document.getElementById('${id}').textContent==='下电'`),'Power feedback did not select off');
  await click(id);const request=await ev(`window.__productionPosts.filter(p=>p.path==='/api/robot/operator').at(-1)`);
  assert(request.body.action===prefix+'_power_off'&&request.body.operation_id==='fixture-owned-power','Off was not bound to owned operation');
  evidence.push({button:id,check:'fenced on/off feedback and operation ownership',request});await pause(700);
 }
 await click('rokaePreview');await ev(`window.renderRokaeStatus({...window.rokaeCurrentStatus,plan:{id:'fixture-plan',state:'ready',next_stage:0,required_confirmation:'go_initial',stages:[{name:'go_initial',label:'到达初始位置'}]},motion_enabled:true,busy:false})`);
 await click('rokaeExecute');
 const move=await ev(`window.__productionPosts.find(p=>p.path==='/api/robot/move')`);
 assert(move?.body.action==='go_initial'&&move.body.confirmation===true&&move.body.motion_token==='present','One-click stage dispatch failed');
 await click('rokaeStop');
 assert(await ev(`document.getElementById('rokaeExecute').disabled`),'Stopped/failed old plan re-enabled');
 evidence.push({check:'one-click stage, failed reply and stop revoke old plan',move});
 assert(errors.length===0,'Browser exception '+JSON.stringify(errors));
 await ev("document.getElementById('taskGate').hidden=true");
 const clip=await ev("(()=>{const r=document.getElementById('rokaeWorkbench').getBoundingClientRect();return {x:r.x+scrollX,y:r.y+scrollY,width:r.width,height:r.height,scale:1}})()");
 const screenshot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true,clip});
 fs.writeFileSync('/tmp/taizhou-production-ui-20261013.png',Buffer.from(screenshot.data,'base64'));
 fs.writeFileSync('/tmp/taizhou-production-browser-20261013.json',JSON.stringify({scope:'deployed frontend, live GET, locally fenced POST; no robot actions',evidence,errors},null,2));
 console.log(JSON.stringify({ok:true,checks:evidence.length,robot_post_forwarded:0,errors}));
}finally{ws.close();}
