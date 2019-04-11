import { Component, OnInit } from '@angular/core';
import { ArrayType } from '@angular/compiler/src/output/output_ast';
import { NnConfigService } from '../../services/nn-config.service';
import { DataExchnageService } from '../../services/data-exchnage.service';
import {Chart} from 'chart.js';


declare var $:any;

@Component({
  selector: 'app-sidebar',
  templateUrl: './sidebar.component.html',
  styleUrls: ['./sidebar.component.css']
})
export class SidebarComponent implements OnInit { 
  public progress:boolean = false;
  LineChart:any;
  hidden1 = false;
  hidden2 = false;
  hidden3 = false;
  f1 :any;
  precision:any;
  recall:any;
  accuracy:any;
  RMSE:any;
  MAE:any;
  MAPE:any;
  RMSPE:any;
  activation = "Tanh" ;
  hiddenlayer = 0;
  learningRate = 0.01;
  epoch = 2 ;
  dataset = "Stock Price";
  neuralNetType = "RNN" ;
  trainTestRatio = 10;
  batchSize = 40;
  hidden1Nodes = 1;
  hidden2Nodes = 1;
  hidden3Nodes = 1;
  outputNodes = 1;
  nnNodes = 1;
  processingStatus = " Idle";
  prediction : any;
  true :any;

  constructor(private nnService:NnConfigService,private dataExchange:DataExchnageService) { 
    
  }
  

  ngOnInit() {
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
    if (val == "Stock Price"){
      this.nnService.changeOutputNodes(1)
      this.outputNodes =1
    }
    else if(val == "20 News Groups"){
      this.nnService.changeOutputNodes(3)
      this.outputNodes =3
    }
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


  sendRNNData(event){
    this.processingStatus = "Processing"
    this.progress = true;

    if (this.LineChart) this.LineChart.destroy();

    this.f1=null;
    this.precision = null
    this.recall = null
    this.accuracy =null
    this.RMSE = null
    this.MAE = null
    this.RMSPE  =null
    this.MAPE = null

    var object = {};
    object['rnnNodes'] = this.nnNodes;
    object['hiddenLayerNum'] = this.hiddenlayer;
    object['hLayer1'] = this.hidden1Nodes;
    object['hLayer2'] = this.hidden2Nodes;
    object['hLayer3'] = this.hidden3Nodes;
    object['outputLayer'] = this.outputNodes;
    object['nnType'] = this.neuralNetType;
    object['learningRate'] = this.learningRate;
    object['dataset'] = this.dataset;
    object['activation'] = this.activation;
    object['epoch'] = this.epoch;
    object['batchSize'] = this.batchSize;
    object['trainTestRatio'] = this.trainTestRatio;

    this.dataExchange.sendPostRequest(object).subscribe(res => {
      this.processingStatus = "Idle"
      this.progress = false;
      this.f1= res.f1;
      this.precision = res.precision
      this.recall = res.recall
      this.accuracy = res.accuracy
      this.RMSE = res.RMSE
      this.MAE = res.MAE
      this.RMSPE  = res.RMSPE
      this.MAPE = res.MAPE
      this.prediction = res.prediction
      this.true = res.true

      console.log(res)

      this.drawChart()   
      });    
  }
  
  

  drawChart(){

    var output:any = [];
    for (var value=0; value<this.true.length; value+=1) {
      output.push(value);
    }

    console.log(this.true)
    console.log(this.prediction)
    console.log(output.length)



    this.LineChart = new Chart('lineChart', {
    type: 'line',
    data: {
     labels:  output,
     datasets: [
     {
      label: 'Prediction',
      data: this.prediction,
      fill:false,
      lineTension:0.01,
      borderColor:"yellow",
      borderWidth: 1
     },
     {
      label: 'Truth',
      data:this.true,
      fill:false,
      lineTension:0.01,
      borderColor:"red",
      borderWidth: 1
  }
    ]
    }, 
    options: {
     title:{
         text:"True Values Vs Predictions",
         display:true
     },
     scales: {
         yAxes: [{
             ticks: {
                 beginAtZero:false
                 
             }
         }]
     }
    }
    });

  }


}
