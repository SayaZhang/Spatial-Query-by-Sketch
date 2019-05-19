export default class Graph{
  constructor(ctx,type,img,pos,edgeColor){
    this.ctx = ctx;
    this.type = type;
    this.img = img;
    this.pos = pos;
    this.edgeColor = edgeColor;
    this.angle = 0;
    this.width = 0;
    this.height = 0;
    this.scaleX = 1;
    this.scaleY = 1;
  }
  draw(){
    this.width = this.img.width;
    this.height = this.img.height;
    this.ctx.save();
    this.ctx.translate(this.pos.x,this.pos.y);
    this.ctx.rotate(this.angle);
    this.ctx.scale(this.scaleX, this.scaleY);
    this.ctx.drawImage(this.img,- this.width / 2, - this.height / 2);
    this.ctx.restore();
  }
  inRange(x,y){
    let points = [
      [this.pos.x - this.width / 2, this.pos.y - this.height / 2],
      [this.pos.x + this.width / 2, this.pos.y - this.height / 2],
      [this.pos.x + this.width / 2, this.pos.y + this.height / 2],
      [this.pos.x - this.width / 2, this.pos.y + this.height / 2]
    ];
    let center = {
      x: this.pos.x,
      y: this.pos.y
    };
    for(let i = 0; i < points.length; i++){
      let x = points[i][0];
      let y = points[i][1];
      points[i][0] = (x - center.x) * Math.cos(this.angle) + (y - center.y) * Math.sin(this.angle) + center.x;
      points[i][1] = -(x - center.x) * Math.sin(this.angle) + (y - center.y) * Math.cos(this.angle) + center.y;
    }
    let inside = false;
    for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
      let xi = points[i][0], yi = points[i][1];
      let xj = points[j][0], yj = points[j][1];
      let intersect = ((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
      if (intersect) inside = !inside;
    }
    return inside;
  }
  move(x,y){
    this.pos.x = x;
    this.pos.y = y;
  }
  rotateSelf(){
    this.angle += Math.PI / 4;
  }
  scaleSelf(scaleX, scaleY){
    this.scaleX = scaleX;
    this.scaleY = scaleY;
  }
  drawEdges(){
    this.ctx.save();
    this.ctx.translate(this.pos.x,this.pos.y);
    this.ctx.rotate(this.angle);
    this.ctx.scale(this.scaleX, this.scaleY);
    this.ctx.strokeStyle = this.edgeColor;
    this.ctx.strokeRect(- this.width / 2 - 5 , - this.height / 2 - 5, this.width + 10 , this.height + 10);
    this.ctx.restore();
  }
}
