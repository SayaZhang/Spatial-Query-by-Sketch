<template>
  <div id="menu">
    <!-- <div class="shapes tool">
      <div class="header" @click="changeShapesSwitch">
        <div class="text">
          <span class="title">Shapes</span>
          <span class="description">(click + drag)</span>
        </div>
        <div class="fold">
          <svg class="icon icon-M fill-dark undefined">
            <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-arrow-down">
              <svg id="icon-arrow-down" viewBox="0 0 32 32" width="100%" height="100%">
                <title>arrow-down</title>
                <path d="M9.88 10.453l6.12 6.107 6.12-6.107 1.88 1.88-8 8-8-8z"></path>
              </svg>
            </use>
          </svg>
        </div>
      </div>
      <div v-show="shapesIsOpen">
        <div class="color-select">
          <template v-if="type === 'coach'">
            <span class="green shapes" @click="changeColor($event)">
              <template v-if="shapesColor === 'green' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
            <span class="blue shapes" @click="changeColor($event)">
              <template v-if="shapesColor === 'blue' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
            <span class="yellow shapes" @click="changeColor($event)">
              <template v-if="shapesColor === 'yellow' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
            <span class="red shapes" @click="changeColor($event)">
              <template v-if="shapesColor === 'red' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
            <span class="black shapes" @click="changeColor($event)">
              <template v-if="shapesColor === 'black' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
            <span class="grey shapes" @click="changeColor($event)">
              <template v-if="shapesColor === 'grey' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
          </template>
          <template v-else>
            <span class="blue shapes" @click="changeColor($event)">
              <template v-if="shapesColor === 'blue' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
            <span class="yellow shapes" @click="changeColor($event)">
              <template v-if="shapesColor === 'yellow' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
            <span class="red shapes" @click="changeColor($event)">
              <template v-if="shapesColor === 'red' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
            <span class="black shapes" @click="changeColor($event)">
              <template v-if="shapesColor === 'black' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
            <span class="grey shapes" @click="changeColor($event)">
              <template v-if="shapesColor === 'grey' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
            <span class="white shapes" @click="changeColor($event)">
              <template v-if="shapesColor === 'white' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
          </template>
        </div>
        <div class="tool-select-three hand">
          <span
            id="square"
            :class="theTool === 'square' ? 'bgGreen' : 'bgWhite'"
            @click.stop="changeTool($event)"
          >
            <svg class="icon icon-M fill-undefined undefined">
              <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-square">
                <svg id="icon-square" viewBox="0 0 32 32" width="100%" height="100%">
                  <title>square</title>
                  <path
                    :fill="theTool === 'square' ? 'white' : color[shapesColor]"
                    d="M24 8v16h-16v-16h16zM26.667 5.333h-21.333v21.333h21.333v-21.333z"
                  ></path>
                </svg>
              </use>
            </svg>
          </span>
          <span
            id="rectangle"
            :class="theTool === 'rectangle' ? 'bgGreen' : 'bgWhite'"
            @click.stop="changeTool($event)"
          >
            <svg class="icon icon-M fill-undefined undefined">
              <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-rectangle">
                <svg id="icon-rectangle" viewBox="0 0 32 32" width="100%" height="100%">
                  <title>rectangle</title>
                  <path
                    :fill="theTool === 'rectangle' ? 'white' : color[shapesColor]"
                    d="M26.667 9.333v13.333h-21.333v-13.333h21.333zM29.333 6.667h-26.667v18.667h26.667v-18.667z"
                  ></path>
                </svg>
              </use>
            </svg>
          </span>
          <span
            id="circular"
            :class="theTool === 'circular' ? 'bgGreen' : 'bgWhite'"
            @click.stop="changeTool($event)"
          >
            <svg class="icon icon-M fill-undefined undefined">
              <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-circle">
                <svg id="icon-circle" viewBox="0 0 32 32" width="100%" height="100%">
                  <title>circle</title>
                  <path
                    :fill="theTool === 'circular' ? 'white' : color[shapesColor]"
                    d="M16 5.333c5.867 0 10.667 4.8 10.667 10.667s-4.8 10.667-10.667 10.667-10.667-4.8-10.667-10.667 4.8-10.667 10.667-10.667zM16 2.667c-7.333 0-13.333 6-13.333 13.333s6 13.333 13.333 13.333 13.333-6 13.333-13.333-6-13.333-13.333-13.333v0z"
                  ></path>
                </svg>
              </use>
            </svg>
          </span>
        </div>
        <div class="tool-select-three-two hand">
          <span
            id="polygon"
            :class="['polygon',theTool === 'polygon' ? 'bgGreen' : 'bgWhite']"
            @click.stop="changeTool($event)"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              version="1.1"
              width="32"
              height="32"
              viewBox="0 0 1697 1024"
            >
              <g id="icomoon-ignore"></g>
              <path
                :stroke="theTool === 'polygon' ? 'white' : color[shapesColor]"
                :fill="theTool === 'polygon' ? '#45d695' : 'white'"
                stroke-width="100"
                stroke-miterlimit="10"
                stroke-linecap="butt"
                stroke-linejoin="miter"
                d="M519.589 18.286h667.429l484.571 493.714-493.714 493.714h-658.286l-493.714-493.714 493.714-493.714z"
              ></path>
            </svg>
          </span>
          <span
            id="reTriangle"
            :class="['polygon',theTool === 'reTriangle' ? 'bgGreen' : 'bgWhite']"
            @click.stop="changeTool($event)"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              version="1.1"
              width="20"
              height="20"
              viewBox="0 0 1024 1024"
            >
              <g id="icomoon-ignore"></g>
              <path
                :stroke="theTool === 'reTriangle' ? 'white' : color[shapesColor]"
                :fill="theTool === 'reTriangle' ? '#45d695' : 'white'"
                stroke-width="100"
                stroke-miterlimit="10"
                stroke-linecap="butt"
                stroke-linejoin="miter"
                d="M506.253 67.211l495.785 889.578-980.076-17.187 484.278-872.391z"
              ></path>
            </svg>
          </span>
          <span class="empty"></span>
        </div>
      </div>
    </div> -->
    <div class="equipment tool">
      <div class="header" @click="changeEquipmentSwitch">
        <div class="text">
          <span class="title">Geo</span>
          <span class="description">(drag)</span>
        </div>
        <!-- <div class="fold">
          <svg class="icon icon-M fill-dark undefined">
            <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-arrow-left">
              <svg id="icon-arrow-left" viewBox="0 0 32 32" width="100%" height="100%">
                <title>arrow-left</title>
                <path d="M9.88 10.453l6.12 6.107 6.12-6.107 1.88 1.88-8 8-8-8z"></path>
              </svg>
            </use>
          </svg>
        </div> -->
      </div>
      <div v-show="equipmentIsOpen">
        <!-- <div class="color-select">
          <template v-if="type === 'coach'">
            <span class="green equipment" @click="changeColor($event)">
              <template v-if="equipmentColor === 'green' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
            <span class="blue equipment" @click="changeColor($event)">
              <template v-if="equipmentColor === 'blue' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
            <span class="yellow equipment" @click="changeColor($event)">
              <template v-if="equipmentColor === 'yellow' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
            <span class="red equipment" @click="changeColor($event)">
              <template v-if="equipmentColor === 'red' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
            <span class="black equipment" @click="changeColor($event)">
              <template v-if="equipmentColor === 'black' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
            <span class="grey equipment" @click="changeColor($event)">
              <template v-if="equipmentColor === 'grey' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
          </template>
          <template v-else>
            <span class="blue equipment" @click="changeColor($event)">
              <template v-if="equipmentColor === 'blue' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
            <span class="yellow equipment" @click="changeColor($event)">
              <template v-if="equipmentColor === 'yellow' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
            <span class="red equipment" @click="changeColor($event)">
              <template v-if="equipmentColor === 'red' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
            <span class="black equipment" @click="changeColor($event)">
              <template v-if="equipmentColor === 'black' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
            <span class="grey equipment" @click="changeColor($event)">
              <template v-if="equipmentColor === 'grey' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
            <span class="white equipment" @click="changeColor($event)">
              <template v-if="equipmentColor === 'white' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
          </template>
        </div> -->
        <div class="tool-select-three move">

          <div 
            v-for="icon in icons"
            :key="icon.type || -1"
            class="select-item"
          >
            <span
            :id="icon.type"
            @mouseover="getBorder($event)"
            @mouseout="removeBorder($event)"
            draggable="true"
            @dragstart="dragStart($event)"
            @drag="drag($event)"
            @dragend="dragEnd($event)"
            >
              <template v-if="graph !== icon.type">
                <svg
                  version="1.1"
                  xmlns="http://www.w3.org/2000/svg"
                  :width="icon.width"
                  :height="icon.height"
                  :viewBox="icon.viewBox"
                >
                <title>{{icon.name}}</title>
                <path v-for="(p,index) in icon.path" :key="index" :d="p" :fill="icon.fill"></path>
                </svg>
              </template>
            </span>
            <label>{{icon.name}}</label>
          </div>
          
        </div>
      </div>
    </div>
    <!-- <div class="players tool">
      <div class="header" @click="changePlayersSwitch">
        <div class="text">
          <span class="title">Players</span>
          <span class="description">(drag)</span>
        </div>
        <div class="fold">
          <svg class="icon icon-M fill-dark undefined">
            <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-arrow-down">
              <svg id="icon-arrow-down" viewBox="0 0 32 32" width="100%" height="100%">
                <title>arrow-down</title>
                <path d="M9.88 10.453l6.12 6.107 6.12-6.107 1.88 1.88-8 8-8-8z"></path>
              </svg>
            </use>
          </svg>
        </div>
      </div>
      <div v-show="playersIsOpen">
        <div class="color-select">
          <template v-if="type === 'coach'">
            <span class="green players" @click="changeColor($event)">
              <template v-if="playersColor === 'green' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
            <span class="blue players" @click="changeColor($event)">
              <template v-if="playersColor === 'blue' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
            <span class="yellow players" @click="changeColor($event)">
              <template v-if="playersColor === 'yellow' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
            <span class="red players" @click="changeColor($event)">
              <template v-if="playersColor === 'red' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
            <span class="black players" @click="changeColor($event)">
              <template v-if="playersColor === 'black' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
            <span class="grey players" @click="changeColor($event)">
              <template v-if="playersColor === 'grey' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
          </template>
          <template v-else>
            <span class="blue players" @click="changeColor($event)">
              <template v-if="playersColor === 'blue' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
            <span class="yellow players" @click="changeColor($event)">
              <template v-if="playersColor === 'yellow' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
            <span class="red players" @click="changeColor($event)">
              <template v-if="playersColor === 'red' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
            <span class="black players" @click="changeColor($event)">
              <template v-if="playersColor === 'black' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
            <span class="grey players" @click="changeColor($event)">
              <template v-if="playersColor === 'grey' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
            <span class="white players" @click="changeColor($event)">
              <template v-if="playersColor === 'white' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
          </template>
        </div>
        <div class="tool-select-two move">
          <span
            id="ring"
            draggable="true"
            @dragstart="dragStart($event)"
            @drag="drag($event)"
            @dragend="dragEnd($event)"
          >
            <template v-if="graph !== 'ring'">
              <svg
                version="1.1"
                xmlns="http://www.w3.org/2000/svg"
                width="20"
                height="20"
                viewBox="0 0 32 32"
              >
                <title>player</title>
                <path
                  fill="white"
                  class="background"
                  d="M30 16c0 7.732-6.268 14-14 14s-14-6.268-14-14c0-7.732 6.268-14 14-14s14 6.268 14 14z"
                ></path>
                <path
                  :fill="color[playersColor]"
                  d="M16 4.8c6.186 0 11.2 5.014 11.2 11.2s-5.014 11.2-11.2 11.2c-6.186 0-11.2-5.014-11.2-11.2v0c0-6.186 5.014-11.2 11.2-11.2v0zM16 0c-8.837 0-16 7.163-16 16s7.163 16 16 16c8.837 0 16-7.163 16-16v0c0-8.837-7.163-16-16-16v0z"
                ></path>
              </svg>
            </template>
          </span>
          <span
            id="halfRing"
            draggable="true"
            @dragstart="dragStart($event)"
            @drag="drag($event)"
            @dragend="dragEnd($event)"
          >
            <template v-if="graph !== 'halfRing'">
              <svg
                version="1.1"
                xmlns="http://www.w3.org/2000/svg"
                width="20"
                height="20"
                viewBox="0 0 32 32"
              >
                <title>goalkeeper</title>
                <path
                  fill="white"
                  class="background"
                  d="M30 16c0 7.732-6.268 14-14 14s-14-6.268-14-14c0-7.732 6.268-14 14-14s14 6.268 14 14z"
                ></path>
                <path
                  :fill="color[playersColor]"
                  d="M28.533 27.467c0.533 0.4 0.8 1.067 0.8 1.867 0 1.467-1.2 2.667-2.667 2.667-0.8 0-1.467-0.267-1.867-0.8-0.4-0.4-0.8-0.8-1.2-1.2-1.733-1.733-2.667-2.667-4-4 1.733-0.667 3.2-1.733 4.4-3.067 1.6-1.867 2.667-4.267 2.667-6.933 0-5.867-4.8-10.667-10.667-10.667s-10.667 4.8-10.667 10.667c0 2.667 1.067 5.067 2.667 6.933 1.2 1.333 2.667 2.4 4.533 3.067-1.333 1.333-2.267 2.267-4 4-0.4 0.4-0.667 0.667-1.2 1.2s-1.2 0.8-2 0.8c-1.467 0-2.667-1.2-2.667-2.667 0-0.8 0.267-1.467 0.8-1.867 0.267-0.267 0.533-0.533 0.667-0.667-2.533-2.933-4.133-6.667-4.133-10.8 0-8.8 7.2-16 16-16s16 7.2 16 16c0 4.133-1.6 7.867-4.133 10.8 0.133 0.133 0.4 0.4 0.667 0.667z"
                ></path>
              </svg>
            </template>
          </span>
        </div>
        <div class="tool-select-two move">
          <span
            id="halfTriangle"
            draggable="true"
            @dragstart="dragStart($event)"
            @drag="drag($event)"
            @dragend="dragEnd($event)"
          >
            <template v-if="graph !== 'halfTriangle'">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                version="1.1"
                width="20"
                height="20"
                viewBox="0 0 1024 1024"
              >
                <g id="icomoon-ignore"></g>
                <path
                  :stroke="theTool === 'halfTriangle' ? 'white' : color[playersColor]"
                  :fill="theTool === 'halfTriangle' ? '#45d695' : 'white'"
                  stroke-width="150"
                  stroke-miterlimit="10"
                  stroke-linecap="butt"
                  stroke-linejoin="miter"
                  d="M813.387 144.624c79.671 78.174 129.056 186.977 129.056 307.32 0 237.72-192.716 430.443-430.443 430.443s-430.443-192.716-430.443-430.443c0-118.872 48.175-226.472 126.080-304.374v0"
                ></path>
                <path
                  :stroke="theTool === 'halfTriangle' ? 'white' : color[playersColor]"
                  :fill="theTool === 'halfTriangle' ? '#45d695' : 'white'"
                  stroke-width="150"
                  stroke-miterlimit="10"
                  stroke-linecap="butt"
                  stroke-linejoin="miter"
                  d="M509.019 448.64l307.027-307.027"
                ></path>
                <path
                  :stroke="theTool === 'halfTriangle' ? 'white' : color[playersColor]"
                  :fill="theTool === 'halfTriangle' ? '#45d695' : 'white'"
                  stroke-width="150"
                  stroke-miterlimit="10"
                  stroke-linecap="butt"
                  stroke-linejoin="miter"
                  d="M509.019 448.64l-301.008-301.008"
                ></path>
              </svg>
            </template>
          </span>
        </div>
      </div>
    </div>
    <div class="lines tool">
      <div class="header" @click="changeLinesSwitch">
        <div class="text">
          <span class="title">Arrows + Lines</span>
          <span class="description">(click + drag)</span>
        </div>
        <div class="fold">
          <svg class="icon icon-M fill-dark undefined">
            <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-arrow-down">
              <svg id="icon-arrow-down" viewBox="0 0 32 32" width="100%" height="100%">
                <title>arrow-down</title>
                <path d="M9.88 10.453l6.12 6.107 6.12-6.107 1.88 1.88-8 8-8-8z"></path>
              </svg>
            </use>
          </svg>
        </div>
      </div>
      <div v-show="linesIsOpen">
        <div class="color-select">
          <template v-if="type === 'coach'">
            <span class="green lines" @click="changeColor($event)">
              <template v-if="linesColor === 'green' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
            <span class="blue lines" @click="changeColor($event)">
              <template v-if="linesColor === 'blue' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
            <span class="yellow lines" @click="changeColor($event)">
              <template v-if="linesColor === 'yellow' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
            <span class="red lines" @click="changeColor($event)">
              <template v-if="linesColor === 'red' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
            <span class="black lines" @click="changeColor($event)">
              <template v-if="linesColor === 'black' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
            <span class="grey lines" @click="changeColor($event)">
              <template v-if="linesColor === 'grey' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
          </template>
          <template v-else>
            <span class="blue lines" @click="changeColor($event)">
              <template v-if="linesColor === 'blue' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
            <span class="yellow lines" @click="changeColor($event)">
              <template v-if="linesColor === 'yellow' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
            <span class="red lines" @click="changeColor($event)">
              <template v-if="linesColor === 'red' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
            <span class="black lines" @click="changeColor($event)">
              <template v-if="linesColor === 'black' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
            <span class="grey lines" @click="changeColor($event)">
              <template v-if="linesColor === 'grey' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
            <span class="white lines" @click="changeColor($event)">
              <template v-if="linesColor === 'white' ">
                <svg class="icon icon-M fill-white undefined">
                  <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-tick">
                    <svg id="icon-tick" viewBox="0 0 32 32" width="100%" height="100%">
                      <title>tick</title>
                      <path
                        fill="white"
                        d="M12 21.6l-5.6-5.6-1.867 1.867 7.467 7.467 16-16-1.867-1.867-14.133 14.133z"
                      ></path>
                    </svg>
                  </use>
                </svg>
              </template>
            </span>
          </template>
        </div>
        <div class="tool-select-two hand">
          <span
            id="solidArrowLine"
            :class="theTool === 'solidArrowLine' ? 'bgGreen' : 'bgWhite'"
            @click="changeTool($event)"
          >
            <svg class="line icon-undefined fill-undefined undefined">
              <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-line-with-arrow">
                <svg id="icon-line-with-arrow" viewBox="0 0 262 32" width="100%" height="100%">
                  <title>line-with-arrow</title>
                  <path
                    :fill="theTool === 'solidArrowLine' ? 'white' : color[linesColor]"
                    d="M256 19.079l6.4-3.079-6.4-3.079v-0.121h-0.251l-22.149-10.656v10.656h-233.6v6.4h233.6v10.656l22.149-10.656h0.251v-0.121z"
                  ></path>
                </svg>
              </use>
            </svg>
          </span>
          <span
            id="dottedArrowLine"
            :class="theTool === 'dottedArrowLine' ? 'bgGreen' : 'bgWhite'"
            @click="changeTool($event)"
          >
            <svg class="line icon-undefined fill-undefined undefined">
              <use
                xmlns:xlink="http://www.w3.org/1999/xlink"
                xlink:href="#icon-dashed-line-with-arrow"
              >
                <svg
                  id="icon-dashed-line-with-arrow"
                  viewBox="0 0 262 32"
                  width="100%"
                  height="100%"
                >
                  <title>dashed-line-with-arrow</title>
                  <path
                    :fill="theTool === 'dottedArrowLine' ? 'white' : color[linesColor]"
                    d="M256 19.079l6.4-3.079-6.4-3.079v-0.121h-0.251l-22.149-10.656v10.656h-11.2v6.4h11.2v10.656l22.149-10.656h0.251v-0.121zM196.8 19.2v-6.4h16v6.4h-16zM171.2 19.2v-6.4h16v6.4h-16zM145.6 19.2v-6.4h16v6.4h-16zM120 19.2v-6.4h16v6.4h-16zM94.4 19.2v-6.4h16v6.4h-16zM68.8 19.2v-6.4h16v6.4h-16zM43.2 19.2v-6.4h16v6.4h-16zM17.6 19.2v-6.4h16v6.4h-16zM8 19.2h-8v-6.4h8v6.4z"
                  ></path>
                </svg>
              </use>
            </svg>
          </span>
        </div>
        <div class="tool-select-two hand">
          <span
            id="waveLine"
            :class="theTool === 'waveLine' ? 'bgGreen' : 'bgWhite'"
            @click="changeTool($event)"
          >
            <svg class="line icon-undefined fill-undefined undefined">
              <use
                xmlns:xlink="http://www.w3.org/1999/xlink"
                xlink:href="#icon-wavy-line-with-arrow"
              >
                <svg id="icon-wavy-line-with-arrow" viewBox="0 0 216 32" width="100%" height="100%">
                  <title>wavy-line-with-arrow</title>
                  <path
                    :fill="theTool === 'waveLine' ? 'white' : color[linesColor]"
                    d="M153.171 27.737c2.193-2.356 4.54-5.929 6.999-10.45 2.398-4.298 4.399-7.357 6.303-9.419 1.862-2.029 3.124-2.521 3.942-2.534 0.793-0.012 2.028 0.419 3.869 2.413 1.9 2.044 3.894 5.119 6.291 9.529l0.758 1.392h10.668v9.76l24-11.547-24-11.547v8h-7.512c-2.218-3.935-4.323-7.066-6.287-9.206-2.25-2.437-4.837-4.172-7.866-4.127-3.005 0.045-5.564 1.831-7.793 4.26-2.185 2.393-4.541 5.969-7.038 10.439l-0.015 0.027c-2.353 4.339-4.318 7.356-6.212 9.363-1.818 1.941-3.079 2.392-3.949 2.38-0.911-0.012-2.256-0.54-4.206-2.54-1.998-2.038-4.118-5.058-6.667-9.298-2.65-4.466-5.133-8.031-7.413-10.417-2.328-2.423-4.948-4.171-7.963-4.214-3.058-0.043-5.665 1.675-7.93 4.129-2.187 2.384-4.527 5.997-6.98 10.573-2.541 4.39-4.648 7.462-6.637 9.506-1.935 2.001-3.248 2.47-4.121 2.458-0.874-0.012-2.157-0.519-4.002-2.531-1.902-2.060-3.879-5.115-6.24-9.418l-0.022-0.039c-2.555-4.473-4.967-8.048-7.201-10.44-2.285-2.434-4.885-4.194-7.91-4.237-3.048-0.043-5.673 1.665-7.98 4.106-2.241 2.385-4.662 6.005-7.221 10.596-2.452 4.408-4.493 7.483-6.438 9.528-1.879 1.989-3.165 2.448-4.028 2.436-0.885-0.013-2.197-0.531-4.098-2.555-1.95-2.064-3.999-5.124-6.453-9.425-2.575-4.588-5.031-8.21-7.323-10.598-2.376-2.46-5.059-4.128-8.158-4.087-3.061 0.040-5.76 1.74-8.199 4.17-2.382 2.385-4.975 5.953-7.743 10.426l4.536 2.805c2.663-4.31 4.88-7.382 6.972-9.454 2.035-2.027 3.474-2.602 4.503-2.615 0.991-0.012 2.349 0.489 4.253 2.46 1.986 2.042 4.047 5.109 6.512 9.51l0.011 0.018c2.555 4.473 4.967 8.048 7.201 10.441 2.286 2.434 4.885 4.193 7.91 4.236 3.048 0.043 5.674-1.665 7.98-4.106 2.241-2.385 4.662-6.004 7.221-10.596 2.453-4.408 4.493-7.483 6.438-9.528 1.88-1.989 3.165-2.448 4.028-2.436 0.885 0.012 2.197 0.53 4.098 2.554 1.949 2.063 3.995 5.12 6.447 9.415 2.455 4.466 4.79 8.043 6.974 10.437 2.243 2.446 4.821 4.217 7.857 4.26 3.037 0.043 5.682-1.655 8.030-4.084 2.288-2.38 4.782-5.991 7.44-10.579l0.023-0.038 0.021-0.040c2.354-4.405 4.322-7.471 6.218-9.511 1.824-1.976 3.082-2.425 3.935-2.413 0.897 0.012 2.237 0.541 4.193 2.576 2.001 2.071 4.124 5.142 6.676 9.45l0.009 0.015c2.653 4.406 5.139 7.924 7.423 10.278 2.333 2.393 4.95 4.108 7.95 4.15 3.043 0.042 5.645-1.642 7.916-4.066z"
                  ></path>
                </svg>
              </use>
            </svg>
          </span>
          <span
            id="dottedLine"
            :class="theTool === 'dottedLine' ? 'bgGreen' : 'bgWhite'"
            @click="changeTool($event)"
          >
            <svg class="line icon-undefined fill-undefined undefined">
              <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-dashed-line">
                <svg id="icon-dashed-line" viewBox="0 0 1200 32" width="100%" height="100%">
                  <title>dashed-line</title>
                  <path
                    :fill="theTool === 'dottedLine' ? 'white' : color[linesColor]"
                    d="M0 32h33.333v-32h-33.333v32zM100 32h66.667v-32h-66.667v32zM233.333 32h66.667v-32h-66.667v32zM366.667 32h66.666v-32h-66.666v32zM500 32h66.667v-32h-66.667v32zM633.333 32h66.667v-32h-66.667v32zM766.667 32h66.666v-32h-66.666v32zM900 32h66.667v-32h-66.667v32zM1033.333 32h66.667v-32h-66.667v32zM1166.667 32h33.333v-32h-33.333v32z"
                  ></path>
                </svg>
              </use>
            </svg>
          </span>
        </div>
      </div>
    </div>
    <div class="text tool">
      <div class="header" @click="changeTextSwitch">
        <div class="text">
          <span class="title">Text + Ruler</span>
          <span class="description">(click + drag)</span>
        </div>
        <div class="fold">
          <svg class="icon icon-M fill-dark undefined">
            <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-arrow-down">
              <svg id="icon-arrow-down" viewBox="0 0 32 32" width="100%" height="100%">
                <title>arrow-down</title>
                <path d="M9.88 10.453l6.12 6.107 6.12-6.107 1.88 1.88-8 8-8-8z"></path>
              </svg>
            </use>
          </svg>
        </div>
      </div>
      <div v-show="textIsOpen">
        <div class="tool-select-two">
          <span
            id="text"
            draggable="true"
            @dragstart="dragStart($event)"
            @drag="drag($event)"
            @dragend="dragEnd($event)"
          >
            <svg class="icon icon-M fill-undefined undefined">
              <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-textbox">
                <svg id="icon-textbox" viewBox="0 0 32 32" width="100%" height="100%">
                  <title>textbox</title>
                  <path
                    d="M30.667 9.333v-8h-8v2.667h-13.333v-2.667h-8v8h2.667v13.333h-2.667v8h8v-2.667h13.333v2.667h8v-8h-2.667v-13.333h2.667zM4 4h2.667v2.667h-2.667v-2.667zM6.667 28h-2.667v-2.667h2.667v2.667zM22.667 25.333h-13.333v-2.667h-2.667v-13.333h2.667v-2.667h13.333v2.667h2.667v13.333h-2.667v2.667zM28 28h-2.667v-2.667h2.667v2.667zM25.333 6.667v-2.667h2.667v2.667h-2.667zM18.307 18.667h-4.653l-0.973 2.667h-2.16l4.533-12h1.867l4.547 12h-2.173l-0.987-2.667zM14.253 16.987h3.48l-1.733-5.107-1.747 5.107z"
                  ></path>
                </svg>
              </use>
            </svg>
          </span>
          <span
            id="ruler"
            :class="['span-hand',theTool === 'ruler' ? 'bgGreen' : 'bgWhite']"
            @click.stop="changeTool($event)"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              version="1.1"
              width="60"
              height="60"
              viewBox="0 0 5178 1024"
            >
              <g id="icomoon-ignore"></g>
              <path
                :fill="theTool === 'ruler' ? 'white' : 'black'"
                d="M3277.28 339.925v344.15h1286.038v307.925l452.831-470.944-470.944-489.056 18.113 362.263z"
              ></path>
              <path
                :fill="theTool === 'ruler' ? 'white' : 'black'"
                d="M1918.793 665.962v-289.812h-1286.038v-344.15l-470.944 489.056 489.056 470.944v-326.038z"
              ></path>
            </svg>
          </span>
        </div>
      </div>
    </div> -->
  </div>
</template>

<script>
export default {
  name: "tools",
  data() {
    return {
      shapesIsOpen: false,
      equipmentIsOpen: true,
      playersIsOpen: false,
      linesIsOpen: false,
      textIsOpen: false,
      graph: ""
    };
  },
  props: ["type"],
  computed: {
    icons() {
      return this.$store.state.iconSvg;
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
    theTool() {
      return this.$store.state.tool;
    }
  },
  methods: {
    changeShapesSwitch() {
      this.shapesIsOpen = !this.shapesIsOpen;
    },
    changeEquipmentSwitch() {
      this.equipmentIsOpen = !this.equipmentIsOpen;
    },
    changePlayersSwitch() {
      this.playersIsOpen = !this.playersIsOpen;
    },
    changeLinesSwitch() {
      this.linesIsOpen = !this.linesIsOpen;
    },
    changeTextSwitch() {
      this.textIsOpen = !this.textIsOpen;
    },
    changeColor(event) {
      let ele = event.target;
      let className = ele.className;
      let theColor = className.split(" ")[0];
      let graph = className.split(" ")[1];
      switch (graph) {
        case "shapes":
          this.$store.commit("setShapesColor", theColor);
          break;
        case "equipment":
          this.$store.commit("setEquipmentColor", theColor);
          break;
        case "players":
          this.$store.commit("setPlayersColor", theColor);
          break;
        case "lines":
          this.$store.commit("setLinesColor", theColor);
          break;
      }
    },
    changeTool(event) {
      let ele = event.target;
      let eleType = ele.nodeName.toLowerCase();
      let theTool;
      if (eleType === "span") {
        theTool = ele.id === this.theTool ? "" : ele.id;
      } else if (eleType === "svg") {
        theTool = ele.parentNode.id === this.theTool ? "" : ele.parentNode.id;
      } else if (eleType === "path") {
        theTool =
          ele.parentNode.parentNode.id === this.theTool
            ? ""
            : ele.parentNode.parentNode.id;
      }
      this.$store.commit("setTool", theTool);
    },
    getBorder() {},
    removeBorder() {},
    dragStart(event) {
      let ele = event.target;
      let eleType = ele.nodeName.toLowerCase();
      let theTool;
      if (eleType === "span") {
        this.graph = ele.id;
        theTool = ele.id === this.theTool ? "" : ele.id;
      } else if (eleType === "svg") {
        this.graph = ele.parentNode.id;
        theTool = ele.parentNode.id === this.theTool ? "" : ele.parentNode.id;
      }
      this.$store.commit("setTool", theTool);
      ele.style["background-color"] = "rgba(255,255,255,0)";
      ele.style["border-color"] = "#e7eaef";
    },
    drag() {},
    dragEnd() {
      this.graph = "";
    }
  },
  mounted() {
    if (this.type === "coach") {
      this.$store.commit("setPlayersColor", "green");
    } else {
      this.$store.commit("setPlayersColor", "red");
    }
  }
};
</script>

<style lang="scss" scoped>
#menu {
  display: flex;
  flex-direction: column;
  background-color: #fafafa;
  border-top: 1px solid #e7eaef;
  border-right: 1px solid #e7eaef;
  width: 100px;
  height: 100%;
  overflow: scroll;
}
.tool {
  padding-left: 10px;
  padding-right: 10px;
}
.move {
  span {
    cursor: move;
  }
}
.hand {
  span {
    cursor: pointer;
  }
}
.span-hand {
  cursor: pointer;
}
.icon {
  width: 24px;
  height: 24px;
}
.line {
  width: 80px;
  height: 15px;
}
.header {
  cursor: pointer;
  justify-content: space-between;
  align-items: center;
  background-color: #fafafa;
  font-size: 12px;
  border-bottom: 1px solid #e7eaef;
  position: fixed;
  .text {
    padding: 6px;
    .title {
      font-size: 0.9rem;   
    }
    .description {
      color: #999;
      font-weight: bold;
    }
  }
  .fold {
    .icon {
      margin-top: 4px;
    }
  }
}
.color-select {
  height: 40px;
  width: 100%;
  font-size: 1rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  span {
    cursor: pointer;
    width: 28px;
    height: 20px;
    border-radius: 2px;
    display: flex;
    justify-content: center;
    align-items: center;
  }
  .green {
    background-color: rgb(69, 214, 149);
  }
  .blue {
    background-color: rgb(63, 149, 220);
  }
  .yellow {
    background-color: rgb(236, 196, 52);
  }
  .red {
    background-color: rgb(229, 79, 58);
  }
  .black {
    background-color: rgb(47, 49, 60);
  }
  .grey {
    background-color: rgb(213, 213, 213);
  }
}
.bgWhite {
  background-color: #fff;
}
.bgGreen {
  background-color: rgb(69, 214, 149);
}
.tool-select-three {
  display: flex;
  justify-content: space-between;
  flex-flow: column;
  align-items: left;
  font-size: 0.6rem;
  overflow: auto;
  margin: 30px 0;
  .select-item{
    width: 100%;
    text-align: center;
    margin-top: 8px;
  }
  span {
    margin: auto;
    border: 1px solid #e7eaef;
    border-radius: 2px;
    width: 65px;
    height: 36px;
    display: flex;
    justify-content: center;
    align-items: center;
    &:hover {
      border-color: #2f313c;
      color: #2f313c;
    }
  }
}
.polygon {
  border: 1px solid #e7eaef;
  border-radius: 2px;
  width: 65px;
  display: flex;
  justify-content: center;
  align-items: center;
  &:hover {
    border-color: #2f313c;
    color: #2f313c;
  }
}
.empty {
  width: 65px;
}
.tool-select-three-two {
  display: flex;
  justify-content: space-between;
  align-items: center;
  height: 50px;
}
.border {
  border-color: #2f313c;
}
.tool-select-two {
  display: flex;
  justify-content: space-between;
  align-items: center;
  height: 50px;
  span {
    border: 1px solid #e7eaef;
    border-radius: 2px;
    color: #2f313c;
    width: 100px;
    height: 34px;
    display: flex;
    justify-content: center;
    align-items: center;
    &:hover {
      border-color: #2f313c;
      color: #2f313c;
    }
  }
}
</style>
