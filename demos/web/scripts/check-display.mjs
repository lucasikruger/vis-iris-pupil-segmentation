import { build } from 'esbuild';
import { readFile,writeFile,mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import cvPromise from '@techstark/opencv-js';
import * as ort from 'onnxruntime-web';
const root=resolve(import.meta.dirname,'../../..'), dir=resolve(process.argv[2]||`${root}/runs/preprocess-parity`), out=resolve(process.argv[3]||`${root}/runs/display-check`);
await mkdir(out,{recursive:true});
async function module(name){const b=await build({entryPoints:[`${root}/demos/web/lib/${name}.ts`],bundle:true,write:false,format:'esm'});return import('data:text/javascript;base64,'+Buffer.from(b.outputFiles[0].text).toString('base64'))}
const pre=await module('preprocess'),{postprocess}=await module('postprocess'),render=await module('render'),display=await module('display');
await pre.initializePreprocess(await cvPromise);ort.env.wasm.numThreads=1;
const session=await ort.InferenceSession.create(new Uint8Array(await readFile(`${root}/demos/web/public/models/res_unet_in_b2__96.onnx`)),{executionProviders:['wasm']});
const manifest=JSON.parse(await readFile(`${dir}/manifest.json`));const rows=[];let batches=[],batch=[[0,0],[0,0]];
for(const {index,stem} of manifest.images){
 const rgb=new Uint8Array(await readFile(`${dir}/${index}.rgb`)),tensor=new ort.Tensor('float32',pre.rgbToTensor(rgb,96),[1,3,96,96]);
 const outputs=await session.run({[session.inputNames[0]]:tensor}),logits=outputs[session.outputNames[0]].data,probs=new Float32Array(27648),argmax=new Uint8Array(9216);
 for(let p=0;p<9216;p++){const max=Math.max(logits[p],logits[9216+p],logits[18432+p]);let sum=0;for(let c=0;c<3;c++){probs[c*9216+p]=Math.exp(logits[c*9216+p]-max);sum+=probs[c*9216+p]};for(let c=0;c<3;c++){probs[c*9216+p]/=sum;if(probs[c*9216+p]>probs[argmax[p]*9216+p])argmax[p]=c}}
 const seg={size:96,numClasses:3,probs,argmax},clean=postprocess(seg);
 await writeFile(`${out}/${index}.raw`,argmax);await writeFile(`${out}/${index}.clean`,clean.argmax);await writeFile(`${out}/${index}.probs`,Buffer.from(probs.buffer));
 const original=argmax.slice();display.displayLabels(seg,96,96);if(!argmax.every((v,i)=>v===original[i]))throw Error('Mutated original labels');
 const target=await readFile(`${dir}/${index}.mask`);for(let p=0;p<9216;p++)for(let c=1;c<3;c++){batch[c-1][0]+=Number(argmax[p]===c&&target[p]===c);batch[c-1][1]+=Number(argmax[p]===c||target[p]===c)}
 if((index+1)%16===0||index===manifest.images.length-1){batches.push(batch.map(([i,u])=>i/(u+1e-6)));batch=[[0,0],[0,0]]}
 const area=a=>a.reduce((s,v)=>s+Number(v>0),0);rows.push({index,stem,raw:area(argmax),clean:area(clean.argmax)});
 if(index<3){await writeFile(`${out}/${index}.resized`,Buffer.from(display.resizeProbabilities(seg,257,193).buffer))}
 tensor.dispose();Object.values(outputs).forEach(o=>o.dispose());
}
const selected=[0,manifest.images.findIndex(r=>r.stem.startsWith('person21_1')),rows.reduce((a,b)=>b.clean-b.raw>a.clean-a.raw?b:a).index];
for(const index of [...new Set(selected)]){
 const bytes=await readFile(`${out}/${index}.probs`);const seg={size:96,numClasses:3,probs:new Float32Array(bytes.buffer,bytes.byteOffset,bytes.length/4).slice(),argmax:new Uint8Array(await readFile(`${out}/${index}.raw`))};
 for(const [name,s] of [['raw',seg],['clean',postprocess(seg)]]){await writeFile(`${out}/${index}.${name}.before`,render.maskPixels(s));await writeFile(`${out}/${index}.${name}.after`,render.displayPixels(s,384,384));}
}
const iou=[0,1].map(c=>batches.reduce((s,b)=>s+b[c],0)/batches.length);
await writeFile(`${out}/manifest.json`,JSON.stringify({images:rows,selected,miou_96:(iou[0]+iou[1])/2},null,2));await session.release();console.log('207 inference maps exported; unchanged 96px mIoU:',(iou[0]+iou[1])/2);
