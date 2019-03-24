import { Component, OnInit } from '@angular/core';
import { ArrayType } from '@angular/compiler/src/output/output_ast';


declare var $:any;

@Component({
  selector: 'app-sidebar',
  templateUrl: './sidebar.component.html',
  styleUrls: ['./sidebar.component.css']
})
export class SidebarComponent implements OnInit { 

  hidden1 = false;
  hidden2 = false;
  hidden3 = false;
  hiddenlayer: number =0 ;
  learningRate: number = 0.01;
  epoch: number= 10;
  dataset: string="Stock Price";
  neuralNetType: string = "RNN";
  trainTestRatio: number = 20;
  batchSize: number = 30;
  hidden1Nodes: number = 1;
  hidden2Nodes : number=1;
  hidden3Nodes:number=1;
  outputNodes = 1;
  nnNodes = 1;
  probtype1 = false;
  probtype2 = true;
  activation : string= "Tanh"
  allNNdata:any = [this.hiddenlayer,this.neuralNetType]

  constructor() { 
    
  }
  

  ngOnInit() {
    $("#menu-toggle").click(function(e) {
      e.preventDefault();
      $("#wrapper").toggleClass("toggled");
    });
  }


  selecHiddenLayers(val: any) {  
    this.hiddenlayer=val
    this.showNumHiddenNodes()
  }
  selecLearningRate(val: any) {  
    this.learningRate=val
  }
  selectEpoch(event: any){
    this.epoch=event.value
  }
  selectDataset(val: any){
    this.dataset=val
    this.showactivation()
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
  selecHidden1Nodes(val: any){
    this.hidden1Nodes = val
  }
  selecHidden2Nodes(val: any){
    this.hidden2Nodes = val    
  }
  selecHidden3Nodes(val: any){
    this.hidden3Nodes = val    
  }
  selectOutputNodes(val:any){
    this.outputNodes = val
  }
  selecnnNodes(val:any){
    this.nnNodes=val
  }
  selectActivation(val:any){
    this.activation=val
  }

  showNumHiddenNodes() {
    if (this.hiddenlayer == 1){
      this.hidden1=true
      this.hidden2=false
      this.hidden3=false
    }
    else if(this.hiddenlayer == 2){
      this.hidden1=true
      this.hidden2=true
      this.hidden3=false
    }
    else if(this.hiddenlayer == 3){
      this.hidden1=true
      this.hidden2=true
      this.hidden3=true
    }
    else if (this.hiddenlayer == 0){
      this.hidden1=false
      this.hidden2=false
      this.hidden3=false
    }
  }
  showactivation(){
    if(this.dataset == "Stock Price"){
      this.probtype1=false
      this.probtype2=true
    }
    else if(this.dataset == "Enron"){
      this.probtype1=true
      this.probtype2=false
    }
  }



}
