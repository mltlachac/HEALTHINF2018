#Last updated November 2017

import pandas as pd
import numpy as np
import math
from sklearn import linear_model
from sklearn.svm import SVR
df = pd.read_csv('dataset.csv', header = 0)
df.head()

#parameters
samples = 20
targetYears = [2015, 2014, 2013]
yearsAhead = 1
totalYears = 9
reportsY = 4
reportsP = 1
priorReports = 1

#calculations
df = df[df["Total Tests (by organism)"] >=samples]
years9 = []
years1 = []
for Y in targetYears:
    years9.append(Y-yearsAhead)
    years1.append(Y-(yearsAhead-1)-totalYears)

#model creation
antibiotic = []
bacteria = []
tY = []
y9 = []
y1 = []

r = []
rY = []
ry9 = []
ry8 = []
ry7 = []
ry6 = []

actualY = []
actual9 = []
actual8 = []
actual7 = []
actual6 = []

sdY = []
sd9 = []
sd8 = []
sd7 = []
sd6 = []

RegLinY = []
RegPolyY = []
SVRlinY = []
SVRrbfY = []

RegLin9 = []
RegPoly9 = []
SVRlin9 = []
SVRrbf9 = []

RegLin8 = []
RegPoly8 = []
SVRlin8 = []
SVRrbf8 = []

LR9e = []
PR9e = []
LS9e = []
GS9e = []
pm9e = []

for org in set(df.organism):
    for ab in set(df.component):
        tdf = df[(df.component == ab) & (df.organism == org)]
        df15 = tdf[(tdf["Report Year"] == 2015)]
        df14 = tdf[(tdf["Report Year"] == 2014)]
        df13 = tdf[(tdf["Report Year"] == 2013)]
        df12 = tdf[(tdf["Report Year"] == 2012)]
        df11 = tdf[(tdf["Report Year"] == 2011)]
        df10 = tdf[(tdf["Report Year"] == 2010)]
        df09 = tdf[(tdf["Report Year"] == 2009)]
        if ((df15.shape[0]>=reportsY) & (df14.shape[0]>=reportsY) & (df13.shape[0]>=reportsY) & (df12.shape[0]>=reportsP) & (df11.shape[0]>=reportsP) & (df10.shape[0]>=reportsP) & (df09.shape[0]>=reportsP)):
            for Y in range(0,len(targetYears)):
                ttdf = tdf[(tdf['Report Year'] != years9[Y]) & (tdf['Report Year'] != targetYears[Y]) & (tdf['Report Year'] >= years1[Y])]
                if ((ttdf.shape[0]>=priorReports)):
                    ttdfY = tdf[(tdf['Report Year'] == targetYears[Y])]
                    ttdf9 = tdf[(tdf['Report Year'] == years9[Y])]
                    ttdf8 = tdf[(tdf['Report Year'] == (years9[Y]-1))]
                    ttdf7 = tdf[(tdf['Report Year'] == (years9[Y]-2))]
                    ttdf6 = tdf[(tdf['Report Year'] == (years9[Y]-3))]
                    ttdf=ttdf.set_index(np.arange(0,ttdf.shape[0]))
                    ttdfY=ttdfY.set_index(np.arange(0,ttdfY.shape[0]))
                    ttdf9=ttdf9.set_index(np.arange(0,ttdf9.shape[0]))
                    ttdf8=ttdf8.set_index(np.arange(0,ttdf8.shape[0]))
                    ttdf7=ttdf7.set_index(np.arange(0,ttdf7.shape[0]))
                    ttdf6=ttdf6.set_index(np.arange(0,ttdf6.shape[0]))       

                    antibiotic.append(ab)
                    bacteria.append(org) 
                    r.append(ttdf.shape[0])
                    rY.append(ttdfY.shape[0])
                    ry9.append(ttdf9.shape[0])
                    ry8.append(ttdf8.shape[0])
                    ry7.append(ttdf7.shape[0])
                    ry6.append(ttdf6.shape[0])
                    tY.append(targetYears[Y])
                    y9.append(years9[Y])
                    y1.append(years1[Y])

                    tempdf9=tdf[(tdf['Report Year'] <= years9[Y]) & (tdf['Report Year'] >= (years1[Y]))]
                    tempdf8=tdf[(tdf['Report Year'] <= (years9[Y]-1)) & (tdf['Report Year'] >= (years1[Y]-1))]
                    tempdf7=tdf[(tdf['Report Year'] <= (years9[Y]-2)) & (tdf['Report Year'] >= (years1[Y]-2))]
                    tempdf9=tempdf9.set_index(np.arange(0,tempdf9.shape[0]))
                    tempdf8=tempdf8.set_index(np.arange(0,tempdf8.shape[0]))
                    tempdf7=tempdf7.set_index(np.arange(0,tempdf7.shape[0]))
                    year9 = []
                    susc9 = []
                    for i in range(0, len(tempdf9)):
                        year9.append(tempdf9["Report Year"][i])
                        susc9.append(tempdf9["Indicator Value (Pct)"][i])
                    year8 = []
                    susc8 = []
                    for i in range(0, len(tempdf8)):
                        year8.append(tempdf8["Report Year"][i])
                        susc8.append(tempdf8["Indicator Value (Pct)"][i])
                    year7 = []
                    susc7 = []
                    for i in range(0, len(tempdf7)):
                        year7.append(tempdf7["Report Year"][i])
                        susc7.append(tempdf7["Indicator Value (Pct)"][i])

                    year9T = np.transpose(np.matrix(year9))
                    susc9T = np.transpose(np.matrix(susc9))
                    year8T = np.transpose(np.matrix(year8))
                    susc8T = np.transpose(np.matrix(susc8))
                    year7T = np.transpose(np.matrix(year7))
                    susc7T = np.transpose(np.matrix(susc7))

                    aY = 0
                    a9 = 0
                    a8 = 0
                    a7 = 0
                    a6 = 0
                    sY = 0
                    s9 = 0
                    s8 = 0
                    s7 = 0       
                    s6 = 0
                    for a in range(0, ttdfY.shape[0]):
                        aY = aY + ttdfY["Total Tests (by organism)"][a] * ttdfY["Indicator Value (Pct)"][a]
                        sY = sY + (a-(aY/sum(ttdfY["Total Tests (by organism)"])))**(2)
                    sdY.append((sY/ttdfY.shape[0])**(1/2))  
                    actualY.append(aY/sum(ttdfY["Total Tests (by organism)"]))
                    for a in range(0, ttdf9.shape[0]):
                        a9 = a9 + ttdf9["Total Tests (by organism)"][a] * ttdf9["Indicator Value (Pct)"][a]
                        s9 = s9 + (a-(a9/sum(ttdf9["Total Tests (by organism)"])))**(2)
                    sd9.append((s9/ttdf9.shape[0])**(1/2))
                    actual9.append(a9/sum(ttdf9["Total Tests (by organism)"]))
                    if (ttdf8.shape[0]>=1):
                        for a in range(0, ttdf8.shape[0]):
                            a8 = a8 + ttdf8["Total Tests (by organism)"][a] * ttdf8["Indicator Value (Pct)"][a]
                            s8 = s8 + (a-(a8/sum(ttdf8["Total Tests (by organism)"])))**(2)
                        sd8.append((s8/ttdf8.shape[0])**(1/2))
                        actual8.append(a8/sum(ttdf8["Total Tests (by organism)"]))
                        regr8 = linear_model.LinearRegression()
                        regr8.fit(year8T, susc8T)
                        RegLin9.append(regr8.predict(years9[Y])[0][0])
                        clfL8 = SVR(kernel = "linear")
                        clfL8.fit(year8T.reshape(-1,1), susc8)
                        SVRlin9.append(clfL8.predict(years9[Y])[0])
                        clf8 = SVR(kernel = "rbf")
                        clf8.fit(year8T.reshape(-1,1), susc8)
                        SVRrbf9.append(clf8.predict(years9[Y])[0])
                        z8 = np.polyfit(year8, susc8, 2)
                        p8 = np.poly1d(z8)
                        RegPoly9.append(p8(years9[Y]))
                    else:
                        actual8.append("NA")
                        sd8.append("NA")
                        RegLin9.append("NA")
                        SVRlin9.append("NA")
                        SVRrbf9.append("NA")
                        RegPoly9.append("NA")
                    if (ttdf7.shape[0]>=1):
                        for a in range(0, ttdf7.shape[0]):
                            a7 = a7 + ttdf7["Total Tests (by organism)"][a] * ttdf7["Indicator Value (Pct)"][a]
                            s7 = s7 + (a-(a7/sum(ttdf7["Total Tests (by organism)"])))**(2)
                        sd7.append((s7/ttdf7.shape[0])**(1/2))
                        actual7.append(a7/sum(ttdf7["Total Tests (by organism)"]))
                    else:
                        actual7.append("NA")
                        sd7.append("NA")
                    if (ttdf6.shape[0]>=1):    
                        for a in range(0, ttdf6.shape[0]):
                            a6 = a6 + ttdf6["Total Tests (by organism)"][a] * ttdf6["Indicator Value (Pct)"][a]
                            s6 = s6 + (a-(a6/sum(ttdf6["Total Tests (by organism)"])))**(2)
                        sd6.append((s6/ttdf6.shape[0])**(1/2))
                        actual6.append(a6/sum(ttdf6["Total Tests (by organism)"]))
                    else:
                        actual6.append("NA")
                        sd6.append("NA")

                    #Linear Regression
                    regr9 = linear_model.LinearRegression()
                    regr9.fit(year9T, susc9T)
                    RegLinY.append(regr9.predict(targetYears[Y])[0][0])

                    #Linear SVR       
                    clfL9 = SVR(kernel = "linear")
                    clfL9.fit(year9T.reshape(-1,1), susc9)
                    SVRlinY.append(clfL9.predict(targetYears[Y])[0])

                    #Gaussian SVR
                    clf9 = SVR(kernel = "rbf")
                    clf9.fit(year9T.reshape(-1,1), susc9)
                    SVRrbfY.append(clf9.predict(targetYears[Y])[0])

                    #Polynomial regression      
                    z9 = np.polyfit(year9, susc9, 2)
                    p9 = np.poly1d(z9)
                    RegPolyY.append(p9(targetYears[Y]))

                    #MSE
                    LR9 = 0
                    LS9 = 0
                    GS9 = 0
                    PR9 = 0
                    pm9 = 0
                    for y in range(0, len(susc9)):
                        LR9 = LR9 + (susc9[y] - regr9.predict(year9[y]))**2
                        LS9 = LS9 + (susc9[y] - clfL9.predict(year9[y]))**2
                        GS9 = GS9 + (susc9[y] - clf9.predict(year9[y]))**2
                        PR9 = PR9 + (susc9[y] - p9(year9[y]))**2
                    LR9e.append(LR9/len(susc9))
                    LS9e.append(LS9/len(susc9))
                    GS9e.append(GS9/len(susc9))
                    PR9e.append(PR9/len(susc9))

#Absolute error
LRYd = []
PRYd = []
LSYd = []
GSYd = []
for c in range(0, len(actualY)):
    LRYd.append(abs(RegLinY[c]-actualY[c]))
    PRYd.append(abs(RegPolyY[c]-actualY[c]))
    LSYd.append(abs(SVRlinY[c]-actualY[c]))
    GSYd.append(abs(SVRrbfY[c]-actualY[c]))
    
LR9d = []
PR9d = []
LS9d = []
GS9d = []
for c in range(0, len(actual9)):
    LR9d.append(abs(RegLin9[c]-actual9[c]))  
    PR9d.append(abs(RegPoly9[c]-actual9[c]))
    LS9d.append(abs(SVRlin9[c]-actual9[c]))
    GS9d.append(abs(SVRrbf9[c]-actual9[c]))
    
pmYd = []
pm9d = []
for c in range(0, len(actualY)):
    pmYd.append(abs(actualY[c]-actual9[c]))
    pm9d.append(abs(actual9[c]-actual8[c]))
                
#Upper bound and PYPER
upperbound = []
upperboundPY = []
upperboundY = []
PYPER = []
PYPERmethod = []
PYPERed = []
PYPERse = []
PYPERsd = []
for i in range(0, len(actualY)):
    upperbound.append(min(LRYd[i], LSYd[i], GSYd[i], PRYd[i]))
    upperboundPY.append(min(LR9d[i], LS9d[i], GS9d[i], PR9d[i]))
    upperboundY.append(min(LR9d[i], LS9d[i], GS9d[i]))
    if (upperboundY[i] == GS9d[i]):
        PYPER.append(SVRrbfY[i])
        PYPERmethod.append("GS")
    elif (upperboundY[i] == LR9d[i]):
        PYPER.append(RegLinY[i])
        PYPERmethod.append("LR")
    elif (upperboundY[i] == LS9d[i]):
        PYPER.append(SVRlinY[i])
        PYPERmethod.append("LS") 
for L in range(0, len(PYPER)):
    rn = ry9[L]
    if pm9d[L]<=10/(rn**(1/2)):
        PYPERed.append(actual9[L])
    else:
        PYPERed.append(PYPER[L])
    if pm9d[L]<=sd9[L]/(rn**(1/2)):
        PYPERse.append(actual9[L])
    else:
        PYPERse.append(PYPER[L])
    if pm9d[L]<=sd9[L]:
        PYPERsd.append(actual9[L])
    else:
        PYPERsd.append(PYPER[L])

#MSE
MSE = []
MSEmethod = []
for i in range(0, len(actualY)):
    minMSE = min(LR9e, LS9e, GS9e)
    minMSEp = min(LR9e, LS9e, GS9e, PR9e)
    if (minMSE[i] == GS9e[i]):
        MSE.append(GSYd[i])
        MSEmethod.append("GS")
    elif (minMSE[i] == LR9e[i]):
        MSE.append(LRYd[i])
        MSEmethod.append("LR")
    elif (minMSE[i] == LS9e[i]):
        MSE.append(LSYd[i])
        MSEmethod.append("LS") 
    
#Adjusted values
PYP = []
for item in PYPER:
    if type(item) == int: 
        if item < 0:
            item = 0
        if item > 100:
            item = 100
    PYP.append(item) 
LRY = []
for item in RegLinY:
    if type(item) == int: 
        if item < 0:
            item = 0
        if item > 100:
            item = 100
    LRY.append(item)  
PRY = []
for item in RegPolyY:
    if type(item) == int: 
        if item < 0:
            print(item)
            item = 0
        if item > 100:
            print(item)
            item = 100
    PRY.append(item)
LSY = []
for item in SVRlinY:
    if type(item) == int: 
        if item < 0:
            item = 0
        if item > 100:
            item = 100
    LSY.append(item)
GSY = []
for item in SVRrbfY:
    if type(item) == int: 
        if item < 0:
            item = 0
        if item > 100:
            item = 100
    GSY.append(item)
Ped = []
for item in PYPERed:
    if type(item) == int: 
        if item < 0:
            item = 0
        if item > 100:
            item = 100
    Ped.append(item)
    
#Absolute error for adjusted values
PYPd = []
LRYd = []
PRYd = []
LSYd = []
GSYd = []
PEDd = []
for c in range(0, len(actualY)):
    PYPd.append(abs(PYP[c]-actualY[c]))    
    LRYd.append(abs(LRY[c]-actualY[c]))
    PRYd.append(abs(PRY[c]-actualY[c]))
    LSYd.append(abs(LSY[c]-actualY[c]))
    GSYd.append(abs(GSY[c]-actualY[c]))
    PEDd.append(abs(Ped[c]-actualY[c]))
    
#Data Frame
outputDF = pd.DataFrame()
outputDF["antibiotic"] = antibiotic
outputDF["bacteria"] = bacteria
outputDF["tY"] = tY
outputDF["y9"] = y9
outputDF["y1"] = y1
outputDF["r"] = r
outputDF["rY"] = rY
outputDF["ry9"] = ry9
outputDF["ry8"] = ry8
outputDF["ry7"] = ry7
outputDF["ry6"] = ry6
outputDF["actualY"] = actualY
outputDF["actual9"] = actual9
outputDF["actual8"] = actual8
outputDF["actual7"] = actual7
outputDF["actual6"] = actual6
outputDF["sdY"] = sdY
outputDF["sd9"] = sd9
outputDF["sd8"] = sd8
outputDF["sd7"] = sd7
outputDF["sd6"] = sd6
outputDF["RegLinY"] = RegLinY
outputDF["RegPolyY"] = RegPolyY
outputDF["SVRlinY"] = SVRlinY
outputDF["SVRrbfY"] = SVRrbfY
outputDF["PYPER"] = PYPER
outputDF["PYPERmethod"] = PYPERmethod
outputDF["upperbound"] = upperbound
outputDF["PYPERed"] = PYPERed
outputDF["PYPERsd"] = PYPERsd
outputDF["PYPERse"] = PYPERse
outputDF["MSE"] = MSE
outputDF["PYPd"] = PYPd
outputDF["LRYd"] = LRYd
outputDF["PRYd"] = PRYd
outputDF["LSYd"] = LSYd
outputDF["GSYd"] = GSYd
outputDF["PEDd"] = PEDd
outputDF["pmYd"] = pmYd

#Evaluation
#for Y in range(0,len(targetYears)):
#indent through end if printing results per year
OutputDf2 = outputDF#[(outputDF.tY == targetYears[Y]) & (outputDF.y9 == years9[Y]) & (outputDF.y1 == years1[Y])]
OutputDf2 = OutputDf2.set_index(np.arange(0,OutputDf2.shape[0]))
methods = []
average = []
thesum = []
methods = ["PYPER", "LinearRegression", "LinearSVR", "GaussianSVR", "PolynomialRegression", "MSE", "upperbound", "PYPERed"]
average = [sum(OutputDf2.PYPd)/len(OutputDf2.PYPd), sum(OutputDf2.LRYd)/len(OutputDf2.LRYd), sum(OutputDf2.LSYd)/len(OutputDf2.LSYd), sum(OutputDf2.GSYd)/len(OutputDf2.GSYd), sum(OutputDf2.PRYd)/len(OutputDf2.PRYd), sum(OutputDf2.MSE)/len(OutputDf2.MSE), sum(OutputDf2.upperbound)/len(OutputDf2.upperbound), sum(OutputDf2.PEDd)/len(OutputDf2.PEDd)]
under5 = []
SET = []

a = 0
b = 0
c = 0
d = 0
e = 0
f = 0
g = 0
h = 0
for i in range(0, len(OutputDf2.PYPER)):
    if OutputDf2.PYPd[i]<=5:
        a = a + 1
    if OutputDf2.LRYd[i]<=5:
        b = b + 1
    if OutputDf2.LSYd[i]<=5:
        c = c + 1
    if OutputDf2.GSYd[i]<=5:
        d = d + 1
    if OutputDf2.PRYd[i]<=5:
        e = e + 1
    if OutputDf2.MSE[i]<=5:
        f = f + 1    
    if OutputDf2.upperbound[i]<=5:
        g = g + 1
    if OutputDf2.PEDd[i]<=5:
        h = h + 1
under5.append(a/len(OutputDf2.PYPER))
under5.append(b/len(OutputDf2.PYPER))
under5.append(c/len(OutputDf2.PYPER))
under5.append(d/len(OutputDf2.PYPER))
under5.append(e/len(OutputDf2.PYPER))
under5.append(f/len(OutputDf2.PYPER))
under5.append(g/len(OutputDf2.PYPER))
under5.append(h/len(OutputDf2.PYPER))

countA = 0
countB = 0
countC = 0
countD = 0
countE = 0
countF = 0
countG = 0
countH = 0
for L in range(0, len(OutputDf2.PYPER)):
    rn = OutputDf2.rY[L]
    if OutputDf2.PYPd[L]<=10/(rn**(1/2)):
        countA = countA + 1
    if OutputDf2.LRYd[L]<=10/(rn**(1/2)):
        countB = countB + 1
    if OutputDf2.LSYd[L]<=10/(rn**(1/2)):
        countC = countC + 1
    if OutputDf2.GSYd[L]<=10/(rn**(1/2)):
        countD = countD + 1
    if OutputDf2.PRYd[L]<=10/(rn**(1/2)):
        countE = countE + 1
    if OutputDf2.MSE[L]<=10/(rn**(1/2)):
        countF = countF + 1    
    if OutputDf2.upperbound[L]<=10/(rn**(1/2)):
        countG = countG + 1
    if OutputDf2.PEDd[L]<=10/(rn**(1/2)):
        countH = countH + 1
SET.append(round(100*countA/len(OutputDf2.PYPER), 2))
SET.append(round(100*countB/len(OutputDf2.PYPER), 2))
SET.append(round(100*countC/len(OutputDf2.PYPER), 2))
SET.append(round(100*countD/len(OutputDf2.PYPER), 2))
SET.append(round(100*countE/len(OutputDf2.PYPER), 2))
SET.append(round(100*countF/len(OutputDf2.PYPER), 2))
SET.append(round(100*countG/len(OutputDf2.PYPER), 2))
SET.append(round(100*countH/len(OutputDf2.PYPER), 2))

#print("Using reports from " + str(years1[Y]) + "-" + str(years9[Y]) + " to predict susceptiblity in " + str(targetYears[Y]))
DF115 = pd.DataFrame()
DF115["method"] = methods
DF115["average"] = average
DF115["under5"] = under5
DF115["SET"] = SET
print(DF115)
print(str(len(OutputDf2.PYPER)/3) + " pairs")





