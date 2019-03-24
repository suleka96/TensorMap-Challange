import { Component, OnInit,Input } from '@angular/core';
import * as d3 from 'd3';

declare var $:any;

@Component({
  selector: 'app-neuralcanvas',
  templateUrl: './neuralcanvas.component.html',
  styleUrls: ['./neuralcanvas.component.css']
})
export class NeuralcanvasComponent implements OnInit {

  @Input() public parentData;

  inputLayerHeight = 4;
  outputLayerHeight=5;
  hiddenLayersDepths =[3,4];
  hiddenLayersCount =2;
  networkGraph :any;
	nodeSize = 15;
  width :any
  height =400
  Math = Math;
  newFirstLayer:any = [];
  hiddenLayers: any =[];
  newGraph :any = {
    "nodes": []
  };

  constructor() { }

  ngOnInit() {
    this.width = $("#neuralNet").innerWidth -15;
    this.draw()
  }

  draw() {
		if (!d3.select("svg")[0]) {
		} else {
			//clear d3
			d3.select('svg').remove();
		}
		var svg = d3.select("#neuralNet").append("svg")
		.attr("width", this.width)
		.attr("height", this.height);

		this.networkGraph = this.buildNodeGraph();
		//buildNodeGraph();
		this.drawGraph(this.networkGraph, svg);
	}


  buildNodeGraph(){
    //construct input layer
		for (var i = 0; i < this.inputLayerHeight; i++) {
			var newTempLayer = {"label": "i"+i, "layer": 1};
			this.newFirstLayer.push(newTempLayer);
    }
    
    //hidden layers
    for (var hiddenLayerLoop = 0; hiddenLayerLoop <this.hiddenLayersCount; hiddenLayerLoop++) {
			var newHiddenLayer:any = [];
			//for the height of this hidden layer
			for (var i = 0; i < this.hiddenLayersDepths[hiddenLayerLoop]; i++) {
				var newTempLayer = {"label": "h"+ hiddenLayerLoop + i, "layer": (hiddenLayerLoop+2)};
				newHiddenLayer.push(newTempLayer);
			}
			this.hiddenLayers.push(newHiddenLayer);
    }

    //construct output layer
		var newOutputLayer:any = [];
		for (var i = 0; i < this.outputLayerHeight; i++) {
			var newoutTempLayer = {"label": "o"+i, "layer": this.hiddenLayersCount + 2};
			newOutputLayer.push(newTempLayer);
    }
    
    //add to newGraph
		var allMiddle = this.newGraph.nodes.concat.apply([], this.hiddenLayers);
		this.newGraph.nodes = this.newGraph.nodes.concat(this.newFirstLayer, allMiddle, newOutputLayer );
  }

  	drawGraph(networkGraph, svg) {
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
    .style("fill", function(d) { return color(d.layer); });
    
    node.append("text")
		.attr("dx", "-.35em")
		.attr("dy", ".35em")
		.attr("font-size", ".6em")
		.text(function(d) { return d.label; });
	}

}
