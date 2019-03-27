import { Component, OnInit } from '@angular/core';
import { ArrayType } from '@angular/compiler/src/output/output_ast';
import { NnConfigService } from '../../services/nn-config.service';

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
  probtype1 = false;
  probtype2 = true;
  activation ;
  hiddenlayer ;
  learningRate;
  epoch;
  dataset;
  neuralNetType ;
  trainTestRatio;
  batchSize;
  hidden1Nodes;
  hidden2Nodes;
  hidden3Nodes;
  outputNodes;
  nnNodes;

  constructor(private nnService:NnConfigService) { 
    
  }
  

  ngOnInit() {
      this.nnService.currenthiddenlayers.subscribe(hidden =>this.hiddenlayer =hidden);
      this.nnService.currentactivation.subscribe(act =>this.activation = act );
      this.nnService.currentbatchSize.subscribe(bs =>this.batchSize = bs );
      this.nnService.currentdataset.subscribe(data => this.dataset= data);
      this.nnService.currentepoch.subscribe(epoch =>this.epoch = epoch);
      this.nnService.currenttrainTestRatio.subscribe(ttr => this.trainTestRatio = ttr );
      this.nnService.currenthidden1Nodes.subscribe(h1 =>this.hidden1Nodes=h1 );
      this.nnService.currenthidden2Nodes.subscribe(h2 =>this.hidden2Nodes= h2);
      this.nnService.currenthidden3Nodes.subscribe(h3 =>this.hidden3Nodes=h3 );
      this.nnService.currentlearningRate.subscribe(lr =>this.learningRate= lr);
      this.nnService.currentneuralNetType.subscribe(nnt => this.neuralNetType=nnt );
      this.nnService.currentoutputNodes.subscribe(out =>this.outputNodes= out);
      this.nnService.currentnnNodes.subscribe(nn => this.nnNodes= nn);   

    $("#menu-toggle").click(function(e) {
      e.preventDefault();
      $("#wrapper").toggleClass("toggled");
    });
  }


  selecHiddenLayers(val: any) {  
    this.hiddenlayer=val
    this.nnService.changeHiddenLayers(val)
    this.showNumHiddenNodes()
  }
  selecLearningRate(val: any) { 
    this.nnService.changeLearningRate(val) 
    this.learningRate=val
  }
  selectEpoch(event: any){
    this.nnService.changeepochs(event.value)
    this.epoch=event.value
  }
  selectDataset(val: any){
    this.nnService.changeDataset(val)
    this.dataset=val
    this.showactivation()
  }
  selectNNType(val: any){
    this.nnService.changeNeuralNetType(val)
    this.neuralNetType=val
  }
  selectTrainTestRatio(event: any){
    this.nnService.changeTrainTestRatio(event.value)
    this.trainTestRatio = event.value
  }
  selectBatchSize(event:any){
    this.nnService.changeBatchSize(event.value)
    this.batchSize = event.value
  }
  selecHidden1Nodes(val: any){
    this.nnService.changeHidden1Nodes(val)
    this.hidden1Nodes = val
  }
  selecHidden2Nodes(val: any){
    this.nnService.changeHidden2Nodes(val)
    this.hidden2Nodes = val    
  }
  selecHidden3Nodes(val: any){
    this.nnService.changeHidden3Nodes(val)
    this.hidden3Nodes = val    
  }
  selectOutputNodes(val:any){
    this.nnService.changeOutputNodes(val)
    this.outputNodes = val
  }
  selecnnNodes(val:any){
    this.nnService.changeNnNodes(val)
    this.nnNodes=val
  }
  selectActivation(val:any){
    this.nnService.changeactivation(val)
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
