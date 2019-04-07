from django.db import models

class RNNConfig(models.Model):
    rnnNodes= models.IntegerField()
    hiddenLayerNum = models.IntegerField()
    hLayer1 = models.IntegerField()
    hLayer2 =models.IntegerField()
    hLayer3 =models.IntegerField()
    outputLayer =models.IntegerField()
    nnType =models.CharField(max_length=4)
    learningRate = models.FloatField()
    dataset =models.CharField(max_length=12)
    activation =models.CharField(max_length=8)
    epoch =models.IntegerField()
    batchSize =models.IntegerField()
    trainTestRatio=models.IntegerField()

class responseConfig(models.Model):
    f1 = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    accuracy = models.FloatField()
    RMSE = models.FloatField()
    MAE = models.FloatField()    
    MAPE = models.FloatField()
    RMSPE = models.FloatField()





