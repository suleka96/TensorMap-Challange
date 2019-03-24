import { Component, OnInit,Input } from '@angular/core';
import * as d3 from 'd3';

@Component({
  selector: 'app-neuralcanvas',
  templateUrl: './neuralcanvas.component.html',
  styleUrls: ['./neuralcanvas.component.css']
})
export class NeuralcanvasComponent implements OnInit {

  @Input() public parentData;

  constructor() { }

  ngOnInit() {
  }

}
