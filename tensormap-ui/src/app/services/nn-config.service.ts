import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs/internal/BehaviorSubject';

@Injectable({
  providedIn: 'root'
})
export class NnConfigService {

  public hiddenlayer =new BehaviorSubject(0)
  currenthiddenlayers = this.hiddenlayer.asObservable();

  public learningRate =new BehaviorSubject(0.01)
  currentlearningRate = this.learningRate.asObservable();

  public epoch =new BehaviorSubject(10)
  currentepoch = this.epoch.asObservable();

  public neuralNetType =new BehaviorSubject("RNN")
  currentneuralNetType = this.neuralNetType.asObservable();

  public dataset =new BehaviorSubject("Stock Price")
  currentdataset = this.dataset.asObservable();

  public trainTestRatio =new BehaviorSubject(20)
  currenttrainTestRatio = this.trainTestRatio.asObservable();

  public batchSize =new BehaviorSubject(30)
  currentbatchSize = this.batchSize.asObservable();

  public hidden1Nodes =new BehaviorSubject(0)
  currenthidden1Nodes = this.hidden1Nodes.asObservable();

  public hidden2Nodes =new BehaviorSubject(0)
  currenthidden2Nodes = this.hidden2Nodes.asObservable();

  public hidden3Nodes =new BehaviorSubject(0)
  currenthidden3Nodes = this.hidden3Nodes.asObservable();

  public outputNodes =new BehaviorSubject(1)
  currentoutputNodes = this.outputNodes.asObservable();

  public nnNodes =new BehaviorSubject(1)
  currentnnNodes = this.nnNodes.asObservable();

  public activation = new BehaviorSubject("Tanh")
  currentactivation = this.activation.asObservable();

  constructor() { }

  changeHiddenLayers(hl: number){
    this.hiddenlayer.next(hl)
  }
  changeLearningRate(lr: number){
    this.learningRate.next(lr)
  }
  changeepochs(epochs: number){
    this.epoch.next(epochs)
  }
  changeNeuralNetType(nnt: string){
    this.neuralNetType.next(nnt)
  }
  changeDataset(data: string){
    this.dataset.next(data)
  }
  changeTrainTestRatio(ttr: number){
    this.trainTestRatio.next(ttr)
  }
  changeBatchSize(bs: number){
    this.batchSize.next(bs)
  }
  changeHidden1Nodes(h1: number){
    this.hidden1Nodes.next(h1)
  }
  changeHidden2Nodes(h2: number){
    this.hidden2Nodes.next(h2)
  }
  changeHidden3Nodes(h3: number){
    this.hidden3Nodes.next(h3)
  }
  changeOutputNodes(oun: number){
    this.outputNodes.next(oun)
  }
  changeNnNodes(nn: number){
    this.nnNodes.next(nn)
  }

  changeactivation(act: string){
    this.activation.next(act)
  }

}
