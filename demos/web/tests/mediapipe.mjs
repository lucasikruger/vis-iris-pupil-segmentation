import { build } from 'esbuild';
import assert from 'node:assert/strict';
const instances=[];
let errors=[];
globalThis.localStorage={getItem:()=>null,setItem:()=>{}};
globalThis.__nativeFactory=async (_,options)=>{
 const item={delegate:options.baseOptions.delegate,closed:false,timestamps:[],
  close(){this.closed=true;},
  detect(){if(errors.shift())throw Error('ROI NaN');return {faceLandmarks:[]};},
  detectForVideo(_,timestamp){this.timestamps.push(timestamp);return this.detect();}};
 instances.push(item);return item;
};
const compiled=await build({entryPoints:['lib/mediapipe.ts'],bundle:true,format:'esm',write:false,plugins:[{
 name:'fake-native',setup(build){build.onResolve({filter:/^@mediapipe\/tasks-vision$/},()=>({path:'native',namespace:'fake'}));build.onLoad({filter:/.*/,namespace:'fake'},()=>({contents:'export const FaceLandmarker={createFromOptions:globalThis.__nativeFactory};export const FilesetResolver={forVisionTasks:async()=>({})};'}));}
}]});
const {ResilientLandmarker}=await import('data:text/javascript;base64,'+Buffer.from(compiled.outputFiles[0].text).toString('base64'));
const detector=new ResilientLandmarker('VIDEO'), video={readyState:2,videoWidth:640,videoHeight:360};
await detector.detectForVideo(video);assert.equal(detector.delegate,'GPU');
errors=[true,false];await detector.detectForVideo(video);
assert.equal(detector.delegate,'CPU');assert(instances[0].closed);
assert.equal(detector.failedFrames,1,'Recovered GPU failure is still counted');
errors=[true,false];await assert.rejects(detector.detectForVideo(video));
assert(instances[1].closed,'Even an isolated CPU error must discard tracking');
await detector.detectForVideo(video);assert.equal(instances.length,3);
errors=[true,true,true,false];
for(let i=0;i<3;i++) { const old=instances.at(-1); await assert.rejects(detector.detectForVideo(video)); if(i===0)assert(old.closed); }
await detector.detectForVideo(video);
assert.equal(detector.failedFrames,5);
assert(instances.slice(0,-1).every(i=>i.closed),'No broken ROI may be reused');
const count=instances.length;
await assert.rejects(detector.detectForVideo({...video,videoWidth:0}));assert.equal(instances.length,count);
const times=instances.flatMap(i=>i.timestamps);assert(times.every((v,i)=>i===0||v>times[i-1]));
const photo=new ResilientLandmarker('IMAGE');await assert.rejects(photo.detect({width:0,height:1}));
await photo.detect({width:640,height:360});assert.equal(photo.delegate,'CPU');
console.log('MediaPipe isolated errors, consecutive GPU/CPU failures, closing, dimensions and timestamps passed');
