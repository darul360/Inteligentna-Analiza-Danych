import random as rand
import math as math
import matplotlib.pyplot as plt

class RBFNeuron:
    waga = 0.0
    sigma = 0.0

class RBF(object):
    def __init__(self, wejscie,wyjscie,numberOfNeurons,wspUczenia,momentum):
        self.wejscie = wejscie
        self.wyjscie = wyjscie
        self.numberOfNeurons = numberOfNeurons
        self.wspUczenia = wspUczenia
        self.momentum = momentum
    #Wczytywanie danych do kolekcji 
    def wczytywanieDanych(self):
        self.daneX = []
        self.daneY = []
        with open('dane.txt', 'r') as f:
            for line in f:
                dane = line.split()  
                self.daneX.append(float(dane[0]))
                self.daneY.append(float(dane[1]))
        f.close()
        self.daneTestX = []
        self.daneTestY = []
        with open('daneTest.txt', 'r') as f:
            for line in f:
                dane = line.split()  
                self.daneTestX.append(float(dane[0]))
                self.daneTestY.append(float(dane[1]))
        f.close()
   #losowanie punktu
    def random_point(self,t):
        return rand.choice(t)
    #odleglosc euklidesowa
    def e_distance(self, point1, point2):
        return abs(point1.waga - point2.waga)
    #uczenie wartswy ukrytej
    def forwardPropagation(self):
        self.wczytywanieDanych()
        self.neuronArrays = []
        self.weights = []
        self.deltaWeights = []
        self.deltaWeights.append(0)
        self.weights.append(rand.uniform(-0.5,0.5))
        for i in range(0, self.numberOfNeurons):
            self.neuronArrays.append(RBFNeuron())
            self.weights.append(rand.uniform(-0.5,0.5))
            self.deltaWeights.append(0)
        for i in range(0, self.numberOfNeurons):
            self.neuronArrays[i].waga = self.random_point(self.daneX)
        for i in range(0, self.numberOfNeurons):
            dis=-1.0
            for j in range(0, self.numberOfNeurons):
                calculatedDistance = self.e_distance(self.neuronArrays[j],self.neuronArrays[i])
                if(i!=j and dis<calculatedDistance):
                    dis=calculatedDistance
            self.neuronArrays[i].sigma=dis/math.sqrt(2*self.numberOfNeurons)
    def gaussianFunction(self, point, neuron):
        return math.exp(-pow(point-neuron.waga,2)/2*neuron.sigma)
    def learn(self,epoka):
        self.calculatedY = [None]*len(self.daneX)
        self.arrayHiddenLayerOutput=[None]*self.numberOfNeurons
        blad=0
        for i in range(0, len(self.daneX)):

            #forward propagation
            calcHiddenLayerVr=0.0
            for j in range (0,self.numberOfNeurons):
                self.arrayHiddenLayerOutput[j]=self.gaussianFunction(self.daneX[i],self.neuronArrays[j])
                calcHiddenLayerVr+=self.arrayHiddenLayerOutput[j]*self.weights[j+1] #suma+=e(wyjscie z wartswy ukrytej)*w
            self.calculatedY[i]=calcHiddenLayerVr+self.weights[0] #bias
            #blad
            blad +=pow(self.calculatedY[i]-self.daneY[i],2)
            #back propagation
            delta=-self.wspUczenia*(self.calculatedY[i]-self.daneY[i])
            self.weights[0]+=delta+self.deltaWeights[0]*self.momentum
            self.deltaWeights[0]=delta
            for j in range (0,self.numberOfNeurons):
                delta1=-self.wspUczenia*(self.calculatedY[i]-self.daneY[i])*self.arrayHiddenLayerOutput[j]
                self.weights[j+1]+=delta1+self.deltaWeights[j+1]*self.momentum
                self.deltaWeights[j+1]=delta1
        return blad/len(self.daneX)
    def compute(self,errorTest):
        self.bladTest = 0
        calculatedVar = [None]*len(self.daneTestX)
        for i in range(0, len(self.daneTestX)):
            calcHiddenLayerVr=0.0
            for j in range (0,self.numberOfNeurons):
                self.arrayHiddenLayerOutput[j]=self.gaussianFunction(self.daneTestX[i],self.neuronArrays[j])
                calcHiddenLayerVr+=self.arrayHiddenLayerOutput[j]*self.weights[j+1]
            calculatedVar[i]=calcHiddenLayerVr+self.weights[0]
            self.bladTest += pow(calculatedVar[i]-self.daneTestY[i],2)
        self.bladTest /= len(self.daneTestX)
        errorTest.append(self.bladTest)
        return calculatedVar

fig = plt.figure()
NN = RBF(1,1,30,0.01,0.9)
NN.forwardPropagation()
line,=plt.plot(NN.daneX,NN.daneY,'ro')
line2,=plt.plot(NN.daneTestX,NN.daneTestY, 'go') 
line1,=plt.plot(NN.daneTestX,NN.daneTestY, 'bo')
errors = []
errorsTest = []
for i in range(0,200):
    errors.append(NN.learn(i))
    print(i)
    line1.set_ydata(NN.compute(errorsTest))
    plt.draw()
    plt.pause(1e-17)
plt.show()
print(errorsTest)

thefile = open('blad.txt', 'w')
for i in range(0,200):
   thefile.write((str(errors[i])+"\n"))
thefile.close()

thefile = open('bladTest.txt', 'w')
for i in range(0,200):
   thefile.write((str(errorsTest[i])+"\n"))
thefile.close()