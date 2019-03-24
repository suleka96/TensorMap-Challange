import { Component, OnInit } from '@angular/core';


declare var $:any;

@Component({
  selector: 'app-sidebar',
  templateUrl: './sidebar.component.html',
  styleUrls: ['./sidebar.component.css']
})
export class SidebarComponent implements OnInit { 

  hiddenlayer: number;
  learningRate: number;
  epoch: number;
  dataset: string;
  neuralNetType: string;
  trainTestRatio: number
  batchSize: number

  constructor() { }
  

  ngOnInit() {
    $("#menu-toggle").click(function(e) {
      e.preventDefault();
      $("#wrapper").toggleClass("toggled");
    });
  }


  selecHiddenLayers(val: any) {  
    this.hiddenlayer=val
  }
  selecLearningRate(val: any) {  
    this.learningRate=val
  }
  selectEpoch(event: any){
    this.epoch=event.value
  }
  selectDataset(val: any){
    this.dataset=val
  }
  selectNNType(val: any){
    this.neuralNetType=val
  }
  selectTrainTestRatio(event: any){
    this.trainTestRatio = event.value
  }
  selectBatchSize(event:any){
    this.batchSize = event.value
  }

}
