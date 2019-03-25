import { Component, OnInit,Input } from '@angular/core';
import {select,schemeCategory10,scaleOrdinal} from 'd3';
import { angularMath } from 'angular-ts-math';
declare var $:any;

@Component({
  selector: 'app-neuralcanvas',
  templateUrl: './neuralcanvas.component.html',
  styleUrls: ['./neuralcanvas.component.css']
})
export class NeuralcanvasComponent implements OnInit {

	// color = scaleOrdinal().range(schemeCategory10)
  inputLayerHeight = 4;
  outputLayerHeight=5;
  hiddenLayersDepths =[3,4];
  hiddenLayersCount =2;
	nodeSize = 17;
  width :any = 500 ;
  height = 400;


  constructor() { }

  ngOnInit() {
    this.draw()
  }

  draw() {
		console.log('in draw')
		if (!select("svg")[0]) {
		} else {
			//clear d3
			select('svg').remove();
		}
		var svg = select("#neuralNet").append("svg")
		.attr("width", this.width)
		.attr("height", this.height);

		var networkGraph : any = this.buildNodeGraph();
		//buildNodeGraph();
		this.drawGraph(networkGraph, svg);
	}


	buildNodeGraph() {
		var newGraph:any = {
			"nodes": []
		};

		//construct input layer
		var newFirstLayer: any = [];
		for (var i = 0; i < this.inputLayerHeight; i++) {
			var newTempLayer1 :any = {"label": "i"+i, "layer": 1};
			newFirstLayer.push(newTempLayer1);
		}

		//construct hidden layers
		var hiddenLayers:any = [];
		for (var hiddenLayerLoop = 0; hiddenLayerLoop < this.hiddenLayersCount; hiddenLayerLoop++) {
			var newHiddenLayer:any = [];
			//for the height of this hidden layer
			for (var i = 0; i < this.hiddenLayersDepths[hiddenLayerLoop]; i++) {
				var newTempLayer2:any = {"label": "h"+ hiddenLayerLoop + i, "layer": (hiddenLayerLoop+2)};
				newHiddenLayer.push(newTempLayer2);
			}
			hiddenLayers.push(newHiddenLayer);
		}

		//construct output layer
		var newOutputLayer:any = [];
		for (var i = 0; i < this.outputLayerHeight; i++) {
			var newTempLayer3 = {"label": "o"+i, "layer": this.hiddenLayersCount + 2};
			newOutputLayer.push(newTempLayer3);
		}

		//add to newGraph
		var allMiddle:any = newGraph.nodes.concat.apply([], hiddenLayers);
		newGraph.nodes = newGraph.nodes.concat(newFirstLayer, allMiddle, newOutputLayer );

		return newGraph;
	}

	drawGraph(networkGraph, svg) {
		var color = scaleOrdinal(schemeCategory10);
		var graph = networkGraph;
		var nodes = graph.nodes;

		// get network size
		var netsize = {};
		nodes.forEach(function (d) {
			if(d.layer in netsize) {
				netsize[d.layer] += 1;
			} else {
				netsize[d.layer] = 1;
			}
			d["lidx"] = netsize[d.layer];
		});

		// calc distances between nodes
		var largestLayerSize = Math.max.apply(
			null, Object.keys(netsize).map(function (i) { return netsize[i]; }));

		var xdist = this.width / Object.keys(netsize).length,
			ydist = (this.height-15) / largestLayerSize;

		// create node locations
		nodes.map(function(d) {
			d["x"] = (d.layer - 0.5) * xdist;
			d["y"] = ( ( (d.lidx - 0.5) + ((largestLayerSize - netsize[d.layer]) /2 ) ) * ydist )+10 ;
		});

		// autogenerate links
		var links:any = [];
		nodes.map(function(d, i) {
			for (var n in nodes) {
				if (d.layer + 1 == nodes[n].layer) {
					links.push({"source": parseInt(i), "target": parseInt(n), "value": 1}) }
			}
		}).filter(function(d) { return typeof d !== "undefined"; });

		// draw links
		var link = svg.selectAll(".link")
		.data(links)
		.enter().append("line")
		.attr("class", "link")
		.attr("x1", function(d) { return nodes[d.source].x; })
		.attr("y1", function(d) { return nodes[d.source].y; })
		.attr("x2", function(d) { return nodes[d.target].x; })
		.attr("y2", function(d) { return nodes[d.target].y; })
		.style("stroke-width", function(d) { return Math.sqrt(d.value); });

		// draw nodes
		var node = svg.selectAll(".node")
		.data(nodes)
		.enter().append("g")
		.attr("transform", function(d) {
			return "translate(" + d.x + "," + d.y + ")"; }
		);

		var circle = node.append("circle")
		.attr("class", "node")
		.attr("r", this.nodeSize)
		.style("fill", function(d) { console.log(color); return color(d.layer); });



		node.append("text")
		.attr("dx", "-.35em")
		.attr("dy", ".35em")
		.attr("font-size", ".6em")
		.text(function(d) { return d.label; });
	}
}
