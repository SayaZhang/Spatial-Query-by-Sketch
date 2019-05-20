<template>
  <div id="ground" ref="ground">
    <div class="field" ref="field">
      <canvas
        id="canvas"
        ref="canvas"
        width="350px"
        height="350px"
        @mousedown="canvasDown($event)"
        @mousemove="canvasMove($event)"
        @mouseup="canvasUp($event)"
        @dragover="dragOver($event)"
        @drop="dragFinished($event)"
      ></canvas>
    </div>
  </div>
</template>

<script>
import Selection from "../tool/selection";
import Polygon from "../tool/polygon";
import Line from "../tool/line";
import Graph from "../tool/graph";
import Icon from "../tool/icon";
import Text from "../tool/text";
export default {
  name: "ground",
  data() {
    return {
      start: {
        x: 0,
        y: 0
      },
      end: {
        x: 0,
        y: 0
      },
      mouseDown: false,
      mouseEnter: false,
      playerStack: [],
      otherStack: [],
      imgWrap: [],
      obj: {},
      selectObj: {},
      imgArr: [],
      geoObjects: [],
      bgImg: {},
      polygonObj: {},
      edgeColor: ""
    };
  },
  props: ["type", "index"],
  computed: {
    tool() {
      return this.$store.state.tool;
    },
    hasGrid() {
      return this.$store.state.hasGrid;
    },
    canvas() {
      return this.$refs.canvas;
    },
    field() {
      return this.$refs.field;
    },
    inputText() {
      return this.$refs.inputText;
    },
    width() {
      return this.$refs.canvas.width;
    },
    height() {
      return this.$refs.canvas.height;
    },
    shapesColor() {
      return this.$store.state.shapesColor;
    },
    equipmentColor() {
      return this.$store.state.equipmentColor;
    },
    playersColor() {
      return this.$store.state.playersColor;
    },
    linesColor() {
      return this.$store.state.linesColor;
    },
    color() {
      return this.$store.state.color;
    },
    downloading() {
      return this.$store.state.downloading;
    },
    toClear() {
      return this.$store.state.toClear;
    },
    toDelete() {
      return this.$store.state.toDelete;
    },
    toRotate() {
      return this.$store.state.toRotate;
    },
    toScale() {
      return this.$store.state.toScale;
    },
    text() {
      return this.$store.state.text;
    },
    icons() {
      return this.$store.state.iconSvg;
    },
    drawObjects: {
      get() {
        return this.$store.state.drawObjects;
      },
      set(value) {
        this.$store.commit("setDrawObjects", value);
      }
    }
  },
  methods: {
    canvasDown(event) {
      let canvas = this.canvas;
      this.start = this.end = this.canvasMousePos(canvas, event);
      this.mouseDown = true;
      let ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, this.width, this.height);
      if (this.tool === "polygon") {
        if (JSON.stringify(this.polygonObj) === "{}") {
          let selectionColor = this.color[this.shapesColor];
          this.polygonObj = new Polygon(
            ctx,
            [[this.start.x, this.start.y]],
            [],
            selectionColor,
            this.edgeColor
          );
        }
      } else {
        this.polygonObj = {};
      }
      let arr = [];
      for (let obj of this.otherStack) {
        obj.draw();
        if (obj.inRange(this.start.x, this.start.y)) {
          arr.push(obj);
        }
      }
      for (let obj of this.playerStack) {
        obj.draw();
        if (obj.inRange(this.start.x, this.start.y)) {
          arr.push(obj);
        }
      }
      let thisObj = {};
      if (arr.length === 1) {
        thisObj = arr[0];
      } else if (arr.length > 1) {
        let selectionArr = [];
        let othersArr = [];
        for (let obj of arr) {
          if (obj instanceof Selection) {
            selectionArr.push(obj);
          } else {
            othersArr.push(obj);
          }
        }
        if (selectionArr.length !== 0 && othersArr.length === 1) {
          thisObj = othersArr[0];
        } else if (othersArr.length === 0) {
          let widthSortedArr = selectionArr.sort(function(a, b) {
            return a.width - b.width;
          });
          let heightSortedArr = selectionArr.sort(function(a, b) {
            return a.height - b.height;
          });
          if (widthSortedArr[0] === heightSortedArr[0]) {
            thisObj = widthSortedArr[0];
          }
        }
      }
      if (JSON.stringify(thisObj) === "{}") {
        this.selectObj = {};
        this.$store.commit("changeSelectState", false);
        this.$store.commit("changePlayerTextState", false);
        this.$store.commit("changeIsLineState", false);
      } else {
        this.$store.commit("changeSelectState", true);
        thisObj.drawEdges();
        this.selectObj = thisObj;
        console.log(this.selectObj);
        if (thisObj instanceof Graph) {
          console.log(
            "graph",
            this.start.x,
            thisObj.pos.x,
            this.start.y,
            thisObj.pos.y
          );
        } else if (thisObj instanceof Selection) {
          thisObj.diffStartX = this.start.x - thisObj.start.x;
          thisObj.diffStartY = this.start.y - thisObj.start.y;
          thisObj.diffEndX = thisObj.end.x - this.start.x;
          thisObj.diffEndY = thisObj.end.y - this.start.y;
          this.$store.commit("changePlayerTextState", false);
          this.$store.commit("changeIsLineState", false);
        } else if (thisObj instanceof Text) {
          thisObj.diffX = this.start.x - thisObj.pos.x;
          thisObj.diffY = this.start.y - thisObj.pos.y;
          this.$store.commit("changePlayerTextState", false);
          this.$store.commit("changeIsLineState", true);
        } else if (thisObj instanceof Icon) {
          this.$store.commit("changeIsLineState", false);
          this.$store.commit("setText", thisObj.text);
          if (
            thisObj.type === "ring" ||
            thisObj.type === "halfTriangle" ||
            thisObj.type === "halfCircular"
          ) {
            this.$store.commit("changeIsPlayerState", false);
            this.$store.commit("changePlayerTextState", true);
          } else if (thisObj.type === "halfRing") {
            this.$store.commit("changeIsPlayerState", true);
            this.$store.commit("changePlayerTextState", true);
          } else {
            this.$store.commit("changePlayerTextState", false);
          }
        } else if (thisObj instanceof Line) {
          if (thisObj.type === "ruler") {
            this.$store.commit("changeIsPlayerState", false);
            this.$store.commit("changePlayerTextState", true);
            this.$store.commit("setText", thisObj.text);
          }
          this.$store.commit("changeIsLineState", true);
        } else if (thisObj instanceof Polygon) {
          for (let point of thisObj.points) {
            let theDiff = [];
            theDiff[0] = this.start.x - point[0];
            theDiff[1] = this.start.y - point[1];
            thisObj.diff.push(theDiff);
          }
          this.$store.commit("changePlayerTextState", false);
          this.$store.commit("changeIsLineState", true);
        } else {
          this.$store.commit("changePlayerTextState", false);
          this.$store.commit("changeIsLineState", false);
        }
      }
    },
    canvasMove(event) {
      let mouseDown = this.mouseDown;
      let tool = this.tool;
      let canvas = this.canvas;
      let ctx = canvas.getContext("2d");
      if (mouseDown) {
        if (this.toScale) {
          this.end = this.canvasMousePos(canvas, event);
          var scaleX =
            1 + (this.end.x - this.selectObj.pos.x) / this.selectObj.width;
          var scaleY =
            1 + (this.end.y - this.selectObj.pos.y) / this.selectObj.height;
          this.selectObj.scaleSelf(scaleX, scaleY);
          ctx.clearRect(0, 0, this.width, this.height);
          this.reDraw();
          this.selectObj.drawEdges();
        } else if (this.tool) {
          this.end = this.canvasMousePos(canvas, event);
          ctx.clearRect(0, 0, this.width, this.height);
          if (
            tool === "square" ||
            tool === "rectangle" ||
            tool === "circular" ||
            tool === "reTriangle"
          ) {
            let selectionColor = this.color[this.shapesColor];
            this.obj = new Selection(
              ctx,
              tool,
              this.start,
              this.end,
              selectionColor,
              this.edgeColor
            );
            this.reDraw();
            this.obj.draw();
            this.obj.drawEdges();
          } else if (tool === "ruler") {
            let selectionColor = this.color[this.shapesColor];
            this.obj = new Line(
              ctx,
              tool,
              this.start,
              this.end,
              selectionColor,
              this.edgeColor
            );
            this.reDraw();
            this.obj.draw();
            this.obj.drawEdges();
          } else if (
            tool === "solidArrowLine" ||
            tool === "dottedArrowLine" ||
            tool === "waveLine" ||
            tool === "dottedLine"
          ) {
            let lineColor = this.color[this.linesColor];
            this.obj = new Line(
              ctx,
              tool,
              this.start,
              this.end,
              lineColor,
              this.edgeColor
            );
            this.reDraw();
            this.obj.draw();
            this.obj.drawEdges();
          } else if (tool === "polygon") {
            this.polygonObj.next = [this.end.x, this.end.y];
            this.reDraw();
            this.polygonObj.draw();
          }
        } else if (JSON.stringify(this.selectObj) !== "{}") {
          this.end = this.canvasMousePos(canvas, event);
          if (this.selectObj instanceof Line) {
            this.selectObj.move(
              this.end.x - this.start.x,
              this.end.y - this.start.y
            );
          } else {
            this.selectObj.move(this.end.x, this.end.y);
          }
          ctx.clearRect(0, 0, this.width, this.height);
          this.reDraw();
          this.selectObj.drawEdges();
        }
      }
    },
    canvasUp() {
      this.mouseDown = false;
      if (JSON.stringify(this.obj) !== "{}") {
        let diffX = Math.abs(this.end.x - this.start.x);
        let diffY = Math.abs(this.end.y - this.start.y);
        let len = Math.sqrt(diffX * diffX + diffY * diffY);
        if (
          ((this.obj.type === "rectangle" || this.obj.type === "circular") &&
            (diffX > 40 || diffY > 40)) ||
          ((this.obj.type === "square" || this.obj.type === "reTriangle") &&
            (diffX > 40 && diffY > 40)) ||
          (this.obj instanceof Line && len > 60)
        ) {
          this.otherStack.push(this.obj);
        } else {
          this.canvas.getContext("2d").clearRect(0, 0, this.width, this.height);
          this.reDraw();
        }
        this.obj = {};
        this.start = {
          x: 0,
          y: 0
        };
        this.end = {
          x: 0,
          y: 0
        };
      } else if (JSON.stringify(this.polygonObj) !== "{}") {
        let len = Math.sqrt(
          Math.pow(this.polygonObj.next[0] - this.polygonObj.points[0][0], 2) +
            Math.pow(this.polygonObj.next[1] - this.polygonObj.points[0][1], 2)
        );
        if (len < 10) {
          this.polygonObj.finish = true;
          this.otherStack.push(this.polygonObj);
          this.canvas.getContext("2d").clearRect(0, 0, this.width, this.height);
          this.reDraw();
          this.polygonObj.drawEdges();
          this.polygonObj = {};
        } else {
          this.polygonObj.points.push(this.polygonObj.next);
        }
        this.start = {
          x: 0,
          y: 0
        };
        this.end = {
          x: 0,
          y: 0
        };
      } else if (this.end.x !== this.start.x || this.end.y !== this.start.y) {
        if (this.selectObj instanceof Line) {
          this.selectObj.cache.start.x = this.selectObj.start.x;
          this.selectObj.cache.start.y = this.selectObj.start.y;
          this.selectObj.cache.end.x = this.selectObj.end.x;
          this.selectObj.cache.end.y = this.selectObj.end.y;
        }
      }
    },
    getScrollTop() {
      let scrollTop = 0;
      if (document.documentElement && document.documentElement.scrollTop) {
        scrollTop = document.documentElement.scrollTop;
      } else if (document.body) {
        scrollTop = document.body.scrollTop;
      }
      return scrollTop;
    },
    canvasMousePos(canvas, event) {
      let x =
        (document.documentElement.scrollLeft || document.body.scrollLeft) +
        (event.clientX || event.pageX);
      let y = (event.clientY || event.pageY) + this.getScrollTop();
      return {
        x: x - canvas.offsetLeft,
        y: y - canvas.offsetTop
      };
    },
    dragOver(event) {
      event.preventDefault();
    },
    dragFinished(event) {
      event.preventDefault();
      let canvas = this.canvas;
      let ctx = canvas.getContext("2d");
      let tool = this.tool;
      //let allGraph = "ball bigGate smallGate wheel railing stool column";
      let allIcon = "point triangle ring halfRing halfTriangle halfCircular";
      this.end = this.canvasMousePos(canvas, event);

      //geoObjects
      if (this.geoObjects.indexOf(tool) > -1) {
        let img = this.imgArr[this.geoObjects.indexOf(tool)];
        this.obj = new Graph(ctx, tool, img, this.end, this.edgeColor);
        ctx.clearRect(0, 0, this.width, this.height);
        this.reDraw();
        this.obj.draw();
        this.obj.drawEdges();
        this.otherStack.push(this.obj);
        this.selectObj = this.obj;
        this.$store.commit("changeSelectState", true);
        this.$store.commit("changePlayerTextState", false);
        this.obj = {};
        this.$store.commit("setTool", "");
      } else if (allIcon.indexOf(tool) > -1) {
        let color = "";
        if (tool === "point" || tool === "triangle") {
          color = this.color[this.equipmentColor];
          this.$store.commit("changePlayerTextState", false);
        } else if (
          tool === "ring" ||
          tool === "halfRing" ||
          tool === "halfTriangle" ||
          tool === "halfCircular"
        ) {
          color = this.color[this.playersColor];
          this.$store.commit("changePlayerTextState", true);
        }
        this.obj = new Icon(ctx, tool, this.end, color, this.edgeColor);
        if (this.tool === "halfRing") {
          this.$store.commit("changeIsPlayerState", true);
          this.$store.commit("setText", "GK");
        } else {
          this.$store.commit("changeIsPlayerState", false);
          this.$store.commit("setText", "");
        }
        ctx.clearRect(0, 0, this.width, this.height);
        this.reDraw();
        this.obj.draw();
        this.obj.drawEdges();
        this.playerStack.push(this.obj);
        this.selectObj = this.obj;
        this.$store.commit("changeSelectState", true);
        this.obj = {};
        this.$store.commit("setTool", "");
      } else if (tool == "text") {
        this.inputText.style.display = "block";
        this.inputText.parentNode.style.position = "relative";
        this.inputText.style.top = this.end.y + "px";
        this.inputText.style.left = this.end.x - 100 + "px";
        this.inputText.focus();
        document.onkeydown = e => {
          if (e.keyCode === 13) {
            let text = this.inputText.value;
            this.inputText.style.display = "none";
            this.inputText.value = "";
            this.inputText.parentNode.style.position = "";
            this.obj = new Text(ctx, text, this.end, this.edgeColor);
            ctx.clearRect(0, 0, this.width, this.height);
            this.reDraw();
            this.obj.draw();
            this.obj.drawEdges();
            this.otherStack.push(this.obj);
            this.selectObj = this.obj;
            this.$store.commit("changeSelectState", true);
            this.$store.commit("changePlayerTextState", false);
            this.obj = {};
            this.$store.commit("setTool", "");
          }
        };
      }
    },
    reDraw() {
      for (let obj of this.otherStack) {
        obj.draw();
      }
      for (let obj of this.playerStack) {
        obj.draw();
      }
    },
    downImg() {
      this.$store.commit("setDrawObjects", this.otherStack);
      this.$ajax({
        url: "http://localhost:5000/search",
        method: "post",
        data: {
          x: JSON.stringify(this.otherStack)
        },
        dataType: "JSON",
        transformRequest: [
          function(data) {
            let ret = "";
            for (let it in data) {
              ret +=
                encodeURIComponent(it) +
                "=" +
                encodeURIComponent(data[it]) +
                "&";
            }
            return ret;
          }
        ]
        // headers: {
        //   "Content-Type": "application/x-www-form-urlencoded"
        // }
      })
        .then(function() {
          console.log("Create!!");
        })
        .catch(function(error) {
          console.log(error);
        });
    }
  },
  watch: {
    downloading() {
      this.downImg();
    },
    toClear() {
      this.otherStack = [];
      this.playerStack = [];
      this.canvas.getContext("2d").clearRect(0, 0, this.width, this.height);
      this.reDraw();
    },
    toDelete() {
      let index1 = this.otherStack.indexOf(this.selectObj);
      let index2 = this.playerStack.indexOf(this.selectObj);
      if (index1 > -1) {
        this.otherStack.splice(index1, 1);
      } else if (index2 > -1) {
        this.playerStack.splice(index2, 1);
      }
      this.selectObj = {};
      this.canvas.getContext("2d").clearRect(0, 0, this.width, this.height);
      this.reDraw();
    },
    toRotate() {
      this.selectObj.rotateSelf();
      this.canvas.getContext("2d").clearRect(0, 0, this.width, this.height);
      this.reDraw();
      this.selectObj.drawEdges();
    },
    text() {
      if (
        this.selectObj instanceof Icon ||
        (this.selectObj instanceof Line && this.selectObj.type === "ruler")
      ) {
        this.selectObj.text = this.text;
        this.canvas.getContext("2d").clearRect(0, 0, this.width, this.height);
        this.reDraw();
        this.selectObj.drawEdges();
      }
    },
    hasGrid() {},
    otherStack() {
      this.$store.commit("setDrawObjects", this.otherStack);
    }
  },
  mounted() {
    let DOMURL = window.URL || window.webkitURL || window;
    var data = [];

    for (var i = 0; i < this.icons.length; i++) {
      var paths = [];
      for (var j = 0; j < this.icons[i].path.length; j++) {
        paths.push(
          `<path fill="` +
            this.icons[i].fill +
            `" d="` +
            this.icons[i].path[j] +
            `"></path>`
        );
      }
      var path = paths.join("");
      this.geoObjects.push(this.icons[i].type);
      data.push(
        `<svg version="1.1" xmlns="http://www.w3.org/2000/svg" width="` +
          this.icons[i].width * 2 +
          `" height="` +
          this.icons[i].height * 2 +
          `" viewBox="` +
          this.icons[i].viewBox +
          `">
              <title>` +
          this.icons[i].name +
          `</title>
              ` +
          path +
          `
        </svg>`
      );
    }

    for (let i = 0; i < data.length; i++) {
      this.imgArr[i] = new Image();
      let svg = new Blob([data[i]], { type: "image/svg+xml;charset=utf-8" });
      this.imgArr[i].src = DOMURL.createObjectURL(svg);
    }

    this.edgeColor = "orange";
    this.$refs.field.style.backgroundImage = `url(/src/assets/grid.bmp)`;
    let index = parseInt(this.index);

    if (index === 1) {
      this.$refs.field.style.backgroundSize = "320px 480px";
      this.$refs.field.style.backgroundPosition = "340px 41px";
    } else if (index === 2) {
      this.$refs.field.style.backgroundSize = "750px 500px";
      this.$refs.field.style.backgroundPosition = "125px 31px";
    } else if (index === 3 || index === 4) {
      this.$refs.field.style.backgroundSize = "400px 480px";
      this.$refs.field.style.backgroundPosition = "300px 41px";
    } else if (index === 5) {
      this.$refs.field.style.backgroundSize = "560px 480px";
      this.$refs.field.style.backgroundPosition = "220px 41px";
    } else if (index === 6 || index === 7) {
      this.$refs.field.style.backgroundSize = "518px 480px";
      this.$refs.field.style.backgroundPosition = "241px 41px";
    } else if (index === 8) {
      this.$refs.field.style.backgroundSize = "442px 480px";
      this.$refs.field.style.backgroundPosition = "279px 41px";
    } else if (index === 9 || index === 10 || index === 11 || index === 16) {
      this.$refs.field.style.backgroundSize = "880px 450px";
      this.$refs.field.style.backgroundPosition = "60px 54px";
    } else if (index === 12) {
      this.$refs.field.style.backgroundSize = "460px 500px";
      this.$refs.field.style.backgroundPosition = "270px 31px";
    } else if (index === 13 || index === 14 || index === 15) {
      this.$refs.field.style.backgroundSize = "500px 500px";
      this.$refs.field.style.backgroundPosition = "250px 31px";
    } else if (index === 17) {
      this.$refs.field.style.backgroundSize = "290px 480px";
      this.$refs.field.style.backgroundPosition = "355px 41px";
    } else if (index === 18) {
      this.$refs.field.style.backgroundSize = "750px 450px";
      this.$refs.field.style.backgroundPosition = "125px 54px";
    }
  }
};
</script>

<style lang="scss" scoped>
#ground {
  background-repeat: repeat;
}
#inputText {
  display: none;
  position: absolute;
  outline: none;
  opacity: 0.5;
  width: 200px;
  height: 20px;
  filter: alpha(opacity=50);
  padding-left: 10px;
  line-height: 20px;
}
.coachInputBorder {
  border: solid 2px rgb(69, 214, 149);
}
.standardInputBorder {
  border: solid 2px white;
}
.field {
  width: 350px;
  height: 350px;
  background-repeat: no-repeat;
  background-size: cover;
  border: 1px solid #333;
  border-right: 0;
}
.coachFieldBorder {
  border: dashed 2px rgb(69, 214, 149);
}
.standardFieldBorder {
  border: dashed 2px white;
}
</style>
