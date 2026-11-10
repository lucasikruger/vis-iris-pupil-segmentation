// Run after PYTHONPATH=. .venv/bin/python scripts/eval/check_preprocess.py.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { build } from 'esbuild';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import cvPromise from '@techstark/opencv-js';
import * as ort from 'onnxruntime-web';
const root = resolve(import.meta.dirname, '../../..');
const dir = resolve(process.argv[2] || `${root}/runs/preprocess-parity`);
const output = resolve(process.argv[3] || `${root}/reports/web-demo/preprocess-parity.json`);
const compiled = await build({entryPoints:[`${root}/demos/web/lib/preprocess.ts`],bundle:true,format:'esm',write:false});
const pre = await import(`data:text/javascript;base64,${Buffer.from(compiled.outputFiles[0].text).toString('base64')}`);
await pre.initializePreprocess(await cvPromise);
ort.env.wasm.numThreads = 1;
const session = await ort.InferenceSession.create(new Uint8Array(await readFile(`${root}/demos/web/public/models/res_unet_in_b2__96.onnx`)),{executionProviders:['wasm']});
const manifest = JSON.parse(await readFile(`${dir}/manifest.json`));
const rows=[], batches=[]; let batch = [[0,0],[0,0]], sum=0, changed=0, maximum=0;
for (const {index,stem} of manifest.images) {
 const rgb=new Uint8Array(await readFile(`${dir}/${index}.rgb`));
 const expected=await readFile(`${dir}/${index}.expected`), actual=pre.preprocessRgb(rgb,96);
 let total=0,count=0,max=0;
 for(let i=0;i<actual.length;i++){const d=Math.abs(actual[i]-expected[i]);total+=d;count+=Number(d!==0);max=Math.max(max,d);}
 sum+=total;changed+=count;maximum=Math.max(maximum,max);rows.push({stem,mae:total/actual.length,changed_channels:count,max_difference:max});
 const tensor=new ort.Tensor('float32',pre.rgbToTensor(rgb,96),[1,3,96,96]);
 const results=await session.run({[session.inputNames[0]]:tensor});const logits=results[session.outputNames[0]].data;
 const target=await readFile(`${dir}/${index}.mask`);
 for(let p=0;p<9216;p++) {let best=0;for(let c=1;c<3;c++)if(logits[c*9216+p]>logits[best*9216+p])best=c;
  for(let c=1;c<3;c++){batch[c-1][0]+=Number(best===c&&target[p]===c);batch[c-1][1]+=Number(best===c||target[p]===c);}}
 tensor.dispose();Object.values(results).forEach(t=>t.dispose());
 if((index+1)%16===0||index===manifest.images.length-1){batches.push(batch.map(([i,u])=>i/(u+1e-6)));batch=[[0,0],[0,0]];}
 if((index+1)%32===0)console.log(`${index+1}/${manifest.images.length}`);
}
await session.release();
const iou=[0,1].map(c=>batches.reduce((s,b)=>s+b[c],0)/batches.length);
const report={model_sha256:createHash('sha256').update(await readFile(`${root}/demos/web/public/models/res_unet_in_b2__96.onnx`)).digest('hex'),n:rows.length,python_opencv:manifest.opencv,web_opencv:'5.0.0',mae:sum/(rows.length*96*96*3),changed_channels:changed,max_difference:maximum,iou_iris:iou[0],iou_pupil:iou[1],miou:(iou[0]+iou[1])/2,aggregation:'Mean of batch IoU, batch size 16; classes 1 and 2; no postprocessing',scope:'Identical decoded/resized RGB crops; does not test browser image decoding or canvas resizing',images:rows};
await mkdir(resolve(output,'..'),{recursive:true});await writeFile(output,JSON.stringify(report,null,2)+'\n');console.log({...report,images:undefined});

assert.equal(rows.length,207);
assert.equal(maximum,0,'Pixel parity failed');
assert.ok(Math.abs(report.miou - .6411911154875957) < .0001,'Webcam mIoU regression');
