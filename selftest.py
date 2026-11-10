import sys, importlib.util
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forge import sdf, post
from forge.render import Camera, render
from forge.sprite import Sprite
from forge.paint import Canvas
from PIL import Image
s = Sprite.from_rows(["...hhhh...","..hhhhhh..","..hvvvvh..","..hkkkkh..","...kkkk...",".pppggppp.","pp.pppp.pp","pp.pppp.pp","...p..p...","..kk..kk.."],
    {"h":"#8f98a3","v":"#35c4d6","k":"#1a1f2b","p":"#586a82","g":"#a8f4ff"})
s.shade("#586a82","#3f4c61").outline("#0b0d12"); a=s.to_image(16)
c = Canvas(480,270,bg="#101820"); c.gradient([(0,"#1b2a49"),(1,"#f2994a")],angle=90)
c.radial(340,150,90,[(0,"#fff3c4"),(1,"#fff3c400")]); c.polygon([(0,270),(140,120),(260,270)],fill="#0d1321")
c.ellipse(100,70,16,fill="#ffffff",opacity=0.8,blur=3); c.rect(300,200,440,240,fill="#e0591c",radius=10)
c.line([(0,230),(480,215)],"#ffffff",width=2); c.text(240,30,"FORGE",size=30,anchor="ma"); c.grain(0.03); b=c.to_image()
sc = sdf.Scene(); sc.material("paint","#7b8794",rough=0.45,coat=0.4,rough_var=0.2); sc.material("metal","#3b4049",metallic=0.8,rough=0.45); sc.material("glow","#3fdcff",emit=4.5)
body = sdf.union(sdf.ellipsoid((0,1.0,0),(0.30,0.40,0.22)), sdf.box((0,0.55,0),(0.22,0.12,0.16),round=0.04), k=0.05)
sc.add("body", body.grooves((0,1,0),[0.9,1.1]), "paint"); sc.add("arm", sdf.capsule((0.34,1.2,0),(0.45,0.7,0),0.07,0.05), "metal", mirror=True); sc.add("eye", sdf.sphere((0,1.15,0.21),0.04), "glow")
r = post.develop(render(sc, Camera(target=(0,0.9,0),yaw=25,pitch=10,height=1.3), 270, 270))
px = post.upscale(post.pixelate(r, 40, colors=10), 6)
out = Image.new("RGB",(1300,290),"#5a5e66"); x=10
for im in (a,b,r,px):
    out.paste(im.convert("RGBA"),(x,10),im.convert("RGBA")); x+=im.width+10
out.save("test.png"); print("ok", a.size,b.size,r.size,px.size)
