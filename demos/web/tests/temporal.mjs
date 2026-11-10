import { build } from 'esbuild';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
async function module(path) {
  const r = await build({ entryPoints: [path], bundle: true, write: false, format: 'esm', platform: 'node' });
  return import('data:text/javascript;base64,' + Buffer.from(r.outputFiles[0].text).toString('base64'));
}
const { TemporalEye, OneEuro } = await module('lib/temporal.ts');
const { postprocess } = await module('lib/postprocess.ts');
const box = { x: 0, y: 0, w: 8, h: 8 };
function seg(labels) {
  const probs = new Float32Array(labels.length * 3);
  labels.forEach((c, i) => { probs[c * labels.length + i] = 1; });
  return { argmax: Uint8Array.from(labels), probs, size: Math.sqrt(labels.length), numClasses: 3 };
}
const eye = new TemporalEye();
let labels = Array(64).fill(1); labels[27] = 2;
eye.box(box, 0); eye.mask(seg(labels), box, 0); eye.center([3.5, 3.5], 0);
const absent = eye.mask(seg(Array(64).fill(1)), box, 33);
assert(!absent.argmax.includes(2), 'Absent pupil must not survive the EMA');
assert.equal(eye.center(null, 33), null);
assert.deepEqual(eye.center([100, 80], 66), [100, 80], 'Reacquisition starts without a stale center');
eye.reset(); assert.deepEqual(eye.box(box, 100), box);
assert.deepEqual(eye.box({ ...box, x: 100 }, 133), { ...box, x: 100 }, 'A large jump resets the box filter');
assert.deepEqual(eye.box({ ...box, x: 101 }, 1600), { ...box, x: 101 }, 'A long gap resets tracking');
// Previous pupil at source x=3.5 maps to pixel x=2.5 after the crop shifts one pixel.
eye.reset(); eye.mask(seg(labels), box, 0);
const shifted = Array(64).fill(1); shifted[26] = 2;
assert.equal(eye.mask(seg(shifted), { ...box, x: 1 }, 33).argmax[26], 2);
assert.equal(eye.mask(seg(shifted), { ...box, x: 1 }, 66).argmax[27], 1);
const euro = new OneEuro(); const input = [], output = [];
for (let i = 0; i < 100; i++) { input.push(50 + (i % 2 ? 1 : -1)); output.push(euro.filter(input.at(-1), i * 33)); }
const jitter = a => a.slice(21).reduce((s, v, i) => s + (v - a[i + 20]) ** 2, 0);
assert(jitter(output) < jitter(input) * .2, 'Filter reduces fixed-target jitter');
const fixtures = JSON.parse(await readFile('tests/morph-fixtures.json', 'utf8'));
for (const f of fixtures) assert.deepEqual([...postprocess(seg(f.input)).argmax], f.output, f.name);
console.log('Temporal reset, absence, alignment, jitter and Python/browser morphology parity passed.');

const { maskPixels } = await module('lib/render.ts');
const labelsContour = Array(32 * 32).fill(0);
for (let y=4;y<28;y++) for(let x=4;x<28;x++) labelsContour[y*32+x]=1;
for (let y=12;y<20;y++) for(let x=12;x<20;x++) labelsContour[y*32+x]=2;
const testMask=seg(labelsContour);
const outline=maskPixels(testMask,'outline',1), thick=maskPixels(testMask,'outline',3);
const alpha=(a,x,y)=>a[(y*32+x)*4+3];
assert.equal(alpha(outline,8,8),0);
assert.equal(alpha(outline,16,16),0);
assert.equal(alpha(outline,4,8),255);
assert.equal(alpha(outline,12,16),255);
assert.equal(alpha(thick,6,8),255);
assert.equal(alpha(maskPixels(testMask,'fill-outline',1),8,8),200);
assert.equal(alpha(maskPixels(testMask,'fill'),0,0),0);
console.log('Mask contour geometry and thickness passed');

const { displayLabels, resizeProbabilities } = await module('lib/display.ts');
const simple = { size: 2, numClasses: 2, argmax: new Uint8Array([0,1,0,1]), probs: new Float32Array([1,0,1,0,0,1,0,1]) };
const beforeLabels=simple.argmax.slice(), beforeProbs=simple.probs.slice();
assert.deepEqual([...displayLabels(simple,4,2)],[0,0,1,1,0,0,1,1]);
assert.deepEqual([...resizeProbabilities(simple,4,2).slice(0,4)],[1,.75,.25,0]);
const corrected={...simple,argmax:new Uint8Array([0,0,0,1])};
assert.deepEqual([...displayLabels(corrected,2,2)],[0,0,0,1]);
assert.deepEqual(simple.argmax,beforeLabels);assert.deepEqual(simple.probs,beforeProbs);
// A destination origin displaced by .5 pixel is mapped through pixel centers.
assert.deepEqual([...displayLabels(simple,5,2,{x:.5,y:0,w:4,h:2})],[0,0,0,1,0,0,0,0,1,0]);
console.log('Display half-pixel interpolation, corrected labels and input immutability passed');

const { displayPixels } = await module('lib/render.ts');
const translucent=displayPixels(testMask,32,32,'fill-outline',2,undefined,.35);
assert.equal(alpha(translucent,8,8),89,'Fill is 35% for iris');
assert.equal(alpha(translucent,16,16),89,'Fill is 35% for pupil');
assert.equal(alpha(translucent,4,8),255,'Outline stays opaque');
const defaultDisplay=displayPixels(testMask,32,32);
assert.equal(alpha(defaultDisplay,8,8),0,'Default is outline only');
assert.equal(alpha(defaultDisplay,5,8),255,'Default outline is 2 display pixels');
assert.equal(alpha(defaultDisplay,6,8),0);
console.log('Default 2px outlines and uniform 35% fill passed');
