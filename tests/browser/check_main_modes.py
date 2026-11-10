"""Exclusive examples/webcam modes; no navigation, no leaked media tracks."""
import argparse,io,json
from pathlib import Path
from PIL import Image
from playwright.sync_api import sync_playwright
parser=argparse.ArgumentParser();parser.add_argument('--video',type=Path,default=Path('runs/webcam.y4m'));args=parser.parse_args()
out=Path('reports/mobile');out.mkdir(parents=True,exist_ok=True);report={}
with sync_playwright() as pw:
 b=pw.chromium.launch(args=['--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader','--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream','--use-file-for-fake-video-capture='+str(args.video.resolve())])
 for name,w,h in [('desktop',1366,768),('mobile',390,844)]:
  ctx=b.new_context(viewport={'width':w,'height':h},device_scale_factor=3 if name=='mobile' else 1,is_mobile=name=='mobile',has_touch=name=='mobile',permissions=['camera'])
  ctx.add_init_script("""localStorage.setItem('iris-mediapipe-delegate','CPU');window.streams=[];window.requests=[];window.sentinel=Math.random();const native=navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);navigator.mediaDevices.getUserMedia=async c=>{window.requests.push(c);const s=await native(c);window.streams.push(s);if(window.delay)await new Promise(r=>window.release=r);return s;}""")
  page=ctx.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)));page.goto('http://127.0.0.1:8765/',wait_until='networkidle')
  sentinel=page.evaluate('window.sentinel');url=page.url
  examples=page.get_by_test_id('examples-mode');webcam=page.get_by_test_id('webcam-mode')
  def fit():
   assert page.evaluate('document.documentElement.scrollWidth===innerWidth')
   assert page.evaluate('window.sentinel')==sentinel and page.url==url
  def shot(suffix):
   Image.open(io.BytesIO(page.screenshot(scale='css'))).convert('RGB').save(out/f'main-modes-{name}-{suffix}.webp',quality=90)
  def processed():examples.get_by_role('status').filter(has_text='1 ojo(s) procesado(s)').wait_for(timeout=120000)
  def start():
   page.get_by_role('button',name='Probar con la cámara',exact=True).click()
   assert webcam.is_visible() and not examples.is_visible()
  def ready():page.wait_for_function("parseFloat(document.querySelector('[data-testid=fps]').textContent)>0",timeout=120000)
  def returned():
   examples.wait_for();assert not webcam.is_visible()
   page.wait_for_function("window.streams.every(s=>s.getTracks().every(t=>t.readyState==='ended')) && !document.querySelector('video').srcObject")
   processed();fit()
  processed();assert page.evaluate('requests.length')==0 and not webcam.is_visible();fit();shot('initial')
  examples.locator('#ejemplos [data-example=reflejo-moderado]').click();processed()
  assert 'moderado' in examples.locator('img[alt="Imagen seleccionada"]').get_attribute('src')
  examples.locator('#resultados').scroll_into_view_if_needed();shot('example')
  start();ready();fit();shot('live')
  assert webcam.get_by_test_id('live-segmented-0').is_visible()
  assert not page.locator('#ejemplos').is_visible()
  page.get_by_role('button',name='Apagar cámara',exact=True).click();returned()
  assert 'moderado' in examples.locator('img[alt="Imagen seleccionada"]').get_attribute('src')
  start();ready();page.get_by_role('button',name='Tomar foto',exact=True).click()
  photo=webcam.get_by_role('region',name='Foto de webcam');photo.get_by_role('status').filter(has_text='Foto procesada').wait_for(timeout=120000)
  assert not examples.is_visible();page.evaluate('scrollTo(0,0)');fit();shot('photo')
  photo.get_by_role('button',name='Volver a ejemplos',exact=True).click();returned()
  # Stopping while acquisition is pending must also close a late-arriving stream.
  page.evaluate('window.delay=true');start();page.wait_for_function('!!window.release')
  page.get_by_role('button',name='Volver a ejemplos',exact=True).click()
  page.evaluate('window.release();window.delay=false');returned()
  assert not errors,errors
  report[name]={'viewport':[w,h],'initial_example_processed_without_camera':True,'exclusive_modes':True,'selected_example_preserved':True,'photo_in_webcam_mode':True,'stop_and_back_release_tracks':True,'late_stream_closed':True,'no_reload_or_navigation':True,'overflow':False,'page_errors':errors}
  ctx.close()
 b.close()
(out/'main-modes.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
