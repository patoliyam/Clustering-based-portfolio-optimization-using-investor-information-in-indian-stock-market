#Clustering based portfolio optimization using investor information in Indian stock market
#By - Patoliya Meetkumar Krushnadas
#e-mail- patoliyameet439@gmail.com


#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import math
import numpy as np
from sklearn.cluster import KMeans
import random
import itertools 

#read from files
df1= pd.read_csv('Old_Format_Of_Share_Holding_Pattern_Top_100_MarCap.csv')
df2= pd.read_csv('Daily_Adjusted_Share_Prices_Top_100_Market_Cap.csv')
df3= pd.read_csv('Monthly_Adjusted_Share_Prices_Top_100_Market_Cap.csv')
df4= pd.read_csv('New_Format_Of_Share_Holding_Pattern_Top_100_MarCap.csv')
df5= pd.read_csv('NIFTY_50_Data.csv')

#Top 100 companies by market cap
Comp = ['Ashok Leyland', 'Asian Paints', 'Bajaj Holdings', 'Bharat Forge',
       'Britannia Inds.', 'Cipla', 'Colgate-Palm.', 'Eicher Motors',
       'Ambuja Cem.', 'Grasim Inds', 'H D F C', 'Hero Motocorp',
       'Hindalco Inds.', 'Hind. Unilever', 'ITC', 'Larsen & Toubro',
       'M & M', 'Bosch', 'MRF', 'Piramal Enterp.', 'Reliance Inds.',
       'P & G Hygiene', 'Vedanta', 'Shree Cement', 'Siemens',
       'Tata Motors', 'Tata Steel', 'Wipro', "Dr Reddy's Labs",
       'Titan Company', 'St Bk of India', 'Shriram Trans.', 'B P C L',
       'Bharat Electron', 'S A I L', 'H P C L', 'B H E L', 'Hind.Zinc',
       'Kotak Mah. Bank', 'UPL', 'Infosys', 'Motherson Sumi', 'Lupin',
       'Zee Entertainmen', 'Pidilite Inds.', 'Havells India',
       'Dabur India', 'Bajaj Fin.', 'Sun Pharma.Inds.', 'Aurobindo Pharma',
       'JSW Steel', 'HDFC Bank', 'TCS', 'ICICI Bank', 'Power Grid Corpn',
       'Bank of Baroda', 'General Insuranc', 'Maruti Suzuki',
       'IndusInd Bank', 'Axis Bank', 'O N G C', 'DLF', 'TVS Motor Co.',
       'United Spirits', 'NTPC', 'I O C L', 'Coal India',
       'Hind.Aeronautics', 'NMDC', 'New India Assura', 'GAIL (India)',
       'Marico', 'Container Corpn.', 'Oracle Fin.Serv.', 'Biocon',
       'Sun TV Network', 'Bharti Airtel', 'Tech Mahindra', 'Petronet LNG',
       "Divi's Lab.", 'Adani Ports', 'Godrej Consumer', 'HDFC Stand. Life',
       'ICICI Pru Life', 'SBI Life Insuran', 'ICICI Lombard',
       'UltraTech Cem.', 'Yes Bank', 'Bajaj Auto', 'Bajaj Finserv',
       'Interglobe Aviat', 'Indiabulls Hous.', 'Aditya Birla Cap',
       'Avenue Super.', 'Bandhan Bank']

#convert date in appropriate format
df1['[Shareholding Date](All)'] = pd.to_datetime(df1['[Shareholding Date](All)'],format='%d/%m/%Y')
df2['[Date](All)'] = pd.to_datetime(df2['[Date](All)'],format='%d/%m/%Y')
df4['[Shareholding date](All)'] = pd.to_datetime(df4['[Shareholding date](All)'],format='%d/%m/%Y')
df5['Date'] = pd.to_datetime(df5['Date'],format='%d %b %Y')

#Sort each company share record by date 
for company in Comp:
    df1.loc[df1['Company Names'] == company].sort_values(['[Shareholding Date](All)'], ascending=[True])

#make array of dates, starting from startDate to <=endDate with period of monthOffset months
def makeDateArray(startDate,monthOffset,endDate):# date format: 'YYYY-MM-DD'
    start = pd.to_datetime(np.datetime64(startDate, dtype='datetime64[D]'))
    end = pd.to_datetime(np.datetime64(endDate, dtype='datetime64[D]'))
    dates = []
    while start<= end:
        dates.append(start)
        start = start+ pd.DateOffset(months=monthOffset)
    return dates

#find cluster of stocks with given period start date and end date (day after last day of period) and given k, which have highest mean TVP
#investorType can be 'Foreign' or 'Institutional'
def StockSelection(startDate,endDate,k,investorType):
    columns = ['Company','Return', 'Foreign','Institutional', 'Individual', 'Volume Ratio']
    Features = pd.DataFrame(columns=columns)
    for company in Comp:
        _df1 =  df1.loc[df1['Company Names'] == company].sort_values(['[Shareholding Date](All)'], ascending=[True])
        _df2 =  df2.loc[df2['Company Names'] == company].sort_values(['[Date](All)'], ascending=[True])
        start = pd.to_datetime(np.datetime64(startDate, dtype='datetime64[D]'))
        end = pd.to_datetime(np.datetime64(endDate, dtype='datetime64[D]'))
        __df1 = _df1.loc[_df1['[Shareholding Date](All)']>=start].loc[_df1['[Shareholding Date](All)']<end]  
        __df2 = _df2.loc[_df2['[Date](All)']>=start].loc[_df2['[Date](All)']<end] 
        
        #no record found in given range of date
        if __df1.shape[0]==0:
            a = _df1.loc[_df1['Company Names'] == company].loc[_df1['[Shareholding Date](All)']<start]
            previous = a.loc[a['[Shareholding Date](All)']==a['[Shareholding Date](All)'].max()]
            b = _df1.loc[_df1['Company Names'] == company].loc[_df1['[Shareholding Date](All)']<previous['[Shareholding Date](All)'].max()]
            previous2 = b.loc[b['[Shareholding Date](All)']==b['[Shareholding Date](All)'].max()]
            __df1 = __df1.append(previous)
            __df1 = __df1.append(previous2)
            __df1 = __df1.sort_values(['[Shareholding Date](All)'], ascending=[True])
        
        #only one record is found in given range of date
        elif __df1.shape[0]==1:
            a = _df1.loc[_df1['Company Names'] == company].loc[_df1['[Shareholding Date](All)']<start]
            previous = a.loc[a['[Shareholding Date](All)']==a['[Shareholding Date](All)'].max()]
            __df1 = __df1.append(previous)
            __df1 = __df1.sort_values(['[Shareholding Date](All)'], ascending=[True])

        if __df1.shape[0]>1 :
            return_ratio = __df2.iloc[[0, -1]]['[Close Price](All)'].diff(1).sum()/__df2.iloc[0]['[Close Price](All)'].sum()
            __df2['Diff'] = __df2['[Close Price](All)'].diff(-1)
            up_volume = __df2.loc[__df2['Diff']>0]['[Total Volume](All)'].sum()
            down_volume = __df2.loc[__df2['Diff']<0]['[Total Volume](All)'].sum()
            volume_ratio = up_volume/float(down_volume) if down_volume>0 else 0
            foreign = __df1['[Total Foreign](All)'].diff(1).abs().sum()
            institutional = __df1['[Total Institutions](All)'].diff(1).abs().sum()
            individual =    __df1['[Total Public & Others](All)'].diff(1).abs().sum()
            govt =    __df1['[Total Govt Holding](All)'].diff(1).abs().sum()
            corporate_holding =  __df1['[Total Non Promoter Corporate Holding](All)'].diff(1).abs().sum()
            promoters =__df1['[Total Promoters](All)'].diff(1).abs().sum()
            total = foreign + institutional + individual + govt + corporate_holding + promoters
            foreign_trade_proportion = (1.0*foreign)/total if total>0 else 0
            institutional_trade_proportion = (1.0*institutional)/total if total>0 else 0
            individual_trade_proportion = (1.0 *individual)/ (total) if total>0 else 0
            Features = Features.append({'Company': company,'Return' : return_ratio ,'Volume Ratio': volume_ratio, 'Foreign': foreign_trade_proportion, 'Institutional' : institutional_trade_proportion, 'Individual':individual_trade_proportion}, ignore_index=True)
    Clustered = pd.DataFrame()
    NameOfFeatures = Features['Company']
    Features = Features.drop(['Company'], axis=1)
    if Features.shape[0]>1 :
        dataset_array = Features.values
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(dataset_array)
        # Get cluster assignment labels
        labels = kmeans.labels_
        # Format results as a DataFrame
        results = pd.DataFrame([Features.index,labels]).T
        #print results
        Features['Cluster Number'] = pd.Series(labels, index=Features.index)
        Features['Company'] = NameOfFeatures
        Clustered=Clustered.append(Features)

    #find mean of every column corresponding to each cluster number
    if len(Clustered.index)>0:
        aggr = Clustered.groupby(['Cluster Number']).mean()
        foreign_cluster_number = aggr.loc[aggr['Foreign'] == aggr['Foreign'].max()].index[0]
        insti_cluster_number = aggr.loc[aggr['Institutional'] == aggr['Institutional'].max()].index[0]
        #cluster_number is the cluster with highest mean TVP  of given type
        cluster_number = foreign_cluster_number if investorType=='Foreign' else insti_cluster_number
        cluster = Clustered.loc[Clustered['Cluster Number'] == cluster_number]
        #stocks in selected cluster
        return list(cluster['Company'])
    return []

# x = StockSelection('2009-09-01','2010-03-01',5,'Institutional') 
# print x

#return weight vector with equal weights assigned to each of the stock
def equalWeightVector(stockList):
    l=len(stockList)
    w = [1.0/l]*l
    return w

#return market capitalization weights known before start of given day
def marketCapitalizationWeightVector(stockList,date_):
    w = []
    _date = pd.to_datetime(np.datetime64(date_, dtype='datetime64[D]'))
    for stock in stockList:
        a = df2.loc[df2['Company Names'] == stock].loc[df2['[Date](All)']<_date]
        b = a.loc[a['[Date](All)']==a['[Date](All)'].max()]
        if(len(b.index)>0):
            marketCap = b.iloc[0]['[Market Cap](All)']
        w.append(marketCap)
    w = np.array(w)/sum(w)
    return w
   
# print marketCapitalizationWeightVector(['Ashok Leyland','Asian Paints'],'2009-01-02')

#find profit(return) in given period with given weight vector for given set of stocks
#w is weight vector assigned to selectedStoks vector
#startDate and endDate are in form of 'YYYY-MM-DD'
#startDate is date of purchase at start of the day, endDate is date of sell at start of day.
def portfolioReturnPercentage(w, selectedStocks, startDate, endDate):
    w = np.array(w)/sum(w)
    start = pd.to_datetime(np.datetime64(startDate, dtype='datetime64[D]'))
    end = pd.to_datetime(np.datetime64(endDate, dtype='datetime64[D]'))
    portfolioReturnPercentage_ = 0
    for i,stock in enumerate(selectedStocks):
        a = df2.loc[df2['Company Names'] == stock].loc[df2['[Date](All)']>=start]
        purchase = a.loc[a['[Date](All)']==a['[Date](All)'].min()]
        #find earliest purchase >=startDate
        purchasePrice = purchase.iloc[0]['[Open Price](All)']
        b = a.loc[a['Company Names'] == stock].loc[a['[Date](All)']<=end]
        #find latest sell <=endDate
        sell = b.loc[b['[Date](All)']==b['[Date](All)'].max()]
        if(len(sell.index)>0):
            sellPrice = sell.iloc[0]['[Open Price](All)']
            portfolioReturnPercentage_ = portfolioReturnPercentage_ + w[i]*(sellPrice - purchasePrice)*100.0/purchasePrice
    return portfolioReturnPercentage_

#nifty50 returns between startDate and endDate in format 'YYYY-MM-DD'
def niftyReturnPercentage(startDate,endDate):
    start = pd.to_datetime(np.datetime64(startDate, dtype='datetime64[D]'))
    end = pd.to_datetime(np.datetime64(endDate, dtype='datetime64[D]'))
    niftyReturnPercentage_ = 0
    a = df5.loc[df5['Date']>=start]
    purchase = a.loc[a['Date']==a['Date'].min()]
    b = df5.loc[df5['Date']<=end]
    sell = b.loc[b['Date']==b['Date'].max()]
    if(len(sell.index)>0):
        purchasePrice = purchase.iloc[0]['Open']
        sellPrice = sell.iloc[0]['Open']
        niftyReturnPercentage_ = (sellPrice - purchasePrice)*100.0/purchasePrice
    return niftyReturnPercentage_

# print portfolioReturnPercentage([0.5,0.5],['Ashok Leyland','Asian Paints'],'2008-01-03','2008-01-06')

#Generate covariance matrix for list of stock stocksList, using historical data between startDate and endDate (in format 'YYYY-MM-DD')
def covarianceMatrix(stocksList,startDate,endDate):
    start = pd.to_datetime(np.datetime64(startDate, dtype='datetime64[D]'))
    end = pd.to_datetime(np.datetime64(endDate, dtype='datetime64[D]'))
    stocks=[]
    for stock in stocksList:
        a = df2.loc[df2['Company Names'] == stock].sort_values(['[Date](All)'], ascending=[True]).loc[df2['[Date](All)']>=start].loc[df2['[Date](All)']<end]
        x = np.array(a['[Open Price](All)'])
        stocks.append(x)
    return np.cov(stocks)
    
# print covarianceMatrix(['Ashok Leyland','Asian Paints'],'2008-01-06','2008-07-01')
    
# lower the portfolio variance, higher the fitness of chromosome, variance is given by w.C.wT
def minVarFitness(chromosome,stocksList,startDate,endDate):
    C = covarianceMatrix(stocksList,startDate,endDate)
    w = np.array(chromosome)/sum(chromosome)
    wDotC=np.dot(w,C)
    variance = np.dot(wDotC,np.array(w).T)
    return -1*variance
    
# higher the sharpe ratop variance, higher the fitness of chromosome, sharpe ratio is given by R_p-R_f/sigma_p
def maxSharpeRatioFitness(chromosome,stocksList,startDate,endDate):
    C = covarianceMatrix(stocksList,startDate,endDate)
    w = np.array(chromosome)/sum(chromosome)
    wDotC=np.dot(w,C)
    variance = np.dot(wDotC,np.array(w).T)
    R_p = portfolioReturnPercentage(w,stocksList,startDate,endDate)
    R_f = 6.50 #risk free rate
    
    return (R_p-R_f)/(100.0*math.sqrt(variance))

def genPop(n,chromGenfuncs, chromGenParams):
#Return a population (list) of N unique individuals.
#Each individual has len(chromgGenFuncs) chromosomes.
#For each individual, chromosome_i is generated by calling chromGenFuncs_i(chromeGenParams_i)
    answer = []
    chromGens = zip(chromGenfuncs, chromGenParams)
    while len(answer) < n:
        indiv = []
        for genfunc, genparams in chromGens:
            indiv.append(float(genfunc(*genparams)))
        answer.append(indiv)
    return list(answer)

#Generate population with n=number of chromosomes, length=length of each chromosome
def generatePopulation(n,length): 
    population = genPop(n,[random.uniform]*length,[(0,1)]*length)
    return population

print generatePopulation(5,2)

#Mutation
#single point mutation with mutation rate = 1/(length of chromosome)
def singlePointMutaion(chromosome):
    mutated = []
    for i in chromosome:
        mutated.append(i)
    geneNumber = random.randint(0, len(chromosome)-1)
    gene = random.uniform(0,1)
    mutated[geneNumber] = gene
    mutated = np.array(mutated)/sum(mutated)
    return mutated

#Crossover
def crossOnes(p1,p2):
    if len(p1)<=1:
        return p1,p2
    else:
        crosspoint = random.randint(1, len(p1)-1)
    child1 = list(itertools.chain(p2[:crosspoint], p1[crosspoint:])) 
    child2 = list(itertools.chain(p1[:crosspoint],p2[crosspoint:]))
    child1 = np.array(child1)/sum(child1)
    child2 = np.array(child2)/sum(child2)
    return child1, child2

#return index of chromosome with highest fitness, scores is array of fitness, where score[i] is fitness of ith chromosome.
def selectHighest(scores):
    maximumIndex = 0
    maximumFitness = scores[0]
    for i,chromosomeFitness in enumerate(scores):
        if(maximumFitness<chromosomeFitness):
            maximumIndex = i
            maximumFitness = chromosomeFitness
    return maximumIndex
            
#return index of chromosome with highest and second highest fitness
def selectTop2(scores):
    maximumIndex = selectHighest(scores)
    minimumIndex = selectLowest(scores)
    maxFitness = scores[maximumIndex] 
    scores[maximumIndex] = scores[minimumIndex]
    secondMaximumIndex = selectHighest(scores)
    scores[maximumIndex] = maxFitness
    return maximumIndex, secondMaximumIndex

#return index of chromosome with lowest fitness
def selectLowest(scores):
    return selectHighest(-1*np.array(scores))

#return index of chromosome with highest and second highest fitness
def selectBottom2(scores):
    return selectTop2(-1*np.array(scores))

#add chromosome in population with rule of survival of fittest
def addOneNewCromosomeToPopulation(newChromosome,population,scores,fitnessFunc, fitnessArgs):
    newScore = fitnessFunc(newChromosome, *fitnessArgs)
    lowIndex = selectLowest(scores)
    lowScore = scores[lowIndex]
    if lowScore<newScore:
        population[lowIndex] = newChromosome
        scores[lowIndex] = newScore
    return

#generations is number of generation till we want to proceed using evolution, totalPopulation is number of chromosomes at any given time
def geneticAlgorithm(totalPopulation,fitnessFunc,stocksList,startDate, endDate, generations):
    population = generatePopulation(totalPopulation,len(stocksList))
    scores = []
    for i,chromosome in enumerate(population):
        scores.append(fitnessFunc(chromosome,stocksList,startDate,endDate))

    for i in xrange(generations):
        action = random.randint(0, len(chromosome)-1)
        # action 0 is for mutation and 1 for crossover
        if action ==0:#mutation
            mutated = singlePointMutaion(population[random.randint(0,len(population)-1)])
            addOneNewCromosomeToPopulation(mutated,population,scores,fitnessFunc,(stocksList,startDate,endDate))
        if action == 1:#crossover
            highIndex1, highIndex2 = selectTop2(scores)
            cross1, cross2 = crossOnes(population[highIndex1],population[highIndex2])
            addOneNewCromosomeToPopulation(cross1,population,scores,fitnessFunc,(stocksList,startDate,endDate))
            addOneNewCromosomeToPopulation(cross2,population,scores,fitnessFunc,(stocksList,startDate,endDate))
    return population[selectHighest(scores)]

# print geneticAlgorithm(5,minVarFitness,['Ashok Leyland','Asian Paints'],'2008-01-01','2008-07-01', 500)

#get fitness function giving highest return, and highest return for given k, investorType and trainingWindowSize (number of months)
def evaluateHyperParameters(k,investorType,trainingWindowSize,startDate, endDate):
    start = pd.to_datetime(np.datetime64(startDate, dtype='datetime64[D]'))
    end = pd.to_datetime(np.datetime64(endDate, dtype='datetime64[D]'))
    equalWeightPortfolioReturn = 1
    marketCapitalizationPortfolioReturn = 1
    minVarianceFitnessPortfolioReturn = 1
    maxSharpeFitnessPortfolioReturn = 1
    NIFTY50Return = 1

    columns = ['Buy Date', 'Sell Date', 'NIFTY50(Day)','NIFTY50(Cummulative)', 'Equal Weight(Day)','Equal Weight(Cummulative)', 'Market Cap(Day)','Market Cap(Cummulative)', 'Min Var(Day)','Min Var(Cummulative)', 'Max Sharpe(Day)','Max Sharpe(Cummulative)']
    returnDf = pd.DataFrame(columns=columns)    
    
    while start<=end:
        #select group of stocks in our portfolio on weekly basis and keep rebalancing portfolio of selected stocks everyday, and repeat till we reach endDate.
        start_ = start.strftime('%Y-%m-%d')
        next_ = (start + pd.DateOffset(days=7)).strftime('%Y-%m-%d')
        trainingWindowStart = (start-pd.DateOffset(months=trainingWindowSize+1)).strftime('%Y-%m-%d')
        trainingWindowEnd = (start - pd.DateOffset(days=1)-pd.DateOffset(months=1)).strftime('%Y-%m-%d')
        testingOfTraingWindowStart = (start -pd.DateOffset(months=1)).strftime('%Y-%m-%d')
        testingOfTraingWindowEnd = (start -pd.DateOffset(days=1)).strftime('%Y-%m-%d')
        
        stocksList = StockSelection(trainingWindowStart,trainingWindowEnd,k,investorType)
        totalPopulation = 50
        generations = 500
        
        rNIFTY = niftyReturnPercentage(start_,next_) #%
        NIFTY50Return =  (1+(rNIFTY/100.0))*NIFTY50Return
        
        wEqWeight = equalWeightVector(stocksList)
        wEqWeight = np.array(wEqWeight)/sum(wEqWeight)
        rEqWeight = portfolioReturnPercentage(wEqWeight, stocksList, start_, next_)
        equalWeightPortfolioReturn = (1+(rEqWeight/100.0))* equalWeightPortfolioReturn
        
        wMarCap = marketCapitalizationWeightVector(stocksList,start_)
        wMarCap = np.array(wMarCap)/sum(wMarCap)
        rMarCap = portfolioReturnPercentage(wMarCap, stocksList, start_, next_)
        marketCapitalizationPortfolioReturn = (1+(rMarCap/100.0))* marketCapitalizationPortfolioReturn
        if rEqWeight!=0:
            wMinVar = geneticAlgorithm(totalPopulation,minVarFitness,stocksList,testingOfTraingWindowStart, testingOfTraingWindowEnd, generations)
            wMinVar = np.array(wMinVar)/sum(wMinVar)
            rMinVar = portfolioReturnPercentage(wMinVar, stocksList, start_, next_)
            minVarianceFitnessPortfolioReturn = (1+(rMinVar/100.0))* minVarianceFitnessPortfolioReturn

            wMaxSharpe = geneticAlgorithm(totalPopulation,maxSharpeRatioFitness,stocksList,testingOfTraingWindowStart, testingOfTraingWindowEnd, generations)
            wMaxSharpe = np.array(wMaxSharpe)/sum(wMaxSharpe)
            rMaxSharpe = portfolioReturnPercentage(wMaxSharpe, stocksList, start_, next_)
            maxSharpeFitnessPortfolioReturn = (1+(rMaxSharpe/100.0))* maxSharpeFitnessPortfolioReturn
        start = start + pd.DateOffset(days=7)

         
    maximumReturn = max(maxSharpeFitnessPortfolioReturn,minVarianceFitnessPortfolioReturn,marketCapitalizationPortfolioReturn,equalWeightPortfolioReturn)
    if (maxSharpeFitnessPortfolioReturn == maximumReturn):
        return maximumReturn, 'maxSharpeRatioFitness'
    elif (minVarianceFitnessPortfolioReturn == maximumReturn):
        return maximumReturn, 'minVarFitness'
    elif (marketCapitalizationPortfolioReturn == maximumReturn):
        return maximumReturn, 'marketCapitalizationWeight'
    else:
        return maximumReturn, 'equalWeight'

#startDate = training period start date, endDate = training period end date
#it returns k, investor type and trainingWindowSize which gives highest return.
def hyperParameterTuningInTrainingPeriod(startDate,endDate): 
    maxReturnK = -1
    maxReturnInvestorType = ''
    maxReturnTrainingWindowSize = 1
    maxReturn = -5000
    for k in range(5,10):
        for investorType in ['Foreign','Institutional']:
            for trainingWindowSize in [3,6,12]:
                print ('Checking hyperparameters- k: ',k,' Investor type:', investorType, ' trainingWindowSize: ',trainingWindowSize)
                start = pd.to_datetime(np.datetime64(startDate, dtype='datetime64[D]'))
                end = pd.to_datetime(np.datetime64(endDate, dtype='datetime64[D]'))
                returnValue, returnFunction = evaluateHyperParameters(k,investorType,trainingWindowSize,startDate, endDate)
                if(returnValue>maxReturn):
                    maxReturn = returnValue
                    maxReturnK = k
                    maxReturnInvestorType = investorType
                    maxReturnTrainingWindowSize = trainingWindowSize
                    maxReturnFunction = returnFunction
                print (' return: ', returnValue, ' returnFunction: ', returnFunction)
    return k,investorType,trainingWindowSize, maxReturnFunction

# print hyperParameterTuningInTrainingPeriod('2009-02-08', '2009-02-11')

# select group of stocks in our portfolio on weekly basis, with given hyper parameters and keep rebalancing portfolio of selected stocks on daily basis. Repeat the process till we reach endDate.
def finalEvaluationInTestingPeriod(totalPopulation,stocksList,startDate, endDate, generations,fitnessFunc,initialNifty,initialOurs):
    start = pd.to_datetime(np.datetime64(startDate, dtype='datetime64[D]'))
    end = pd.to_datetime(np.datetime64(endDate, dtype='datetime64[D]'))
    equalWeightPortfolioReturn = initialOurs
    marketCapitalizationPortfolioReturn = initialOurs
    minVarianceFitnessPortfolioReturn = initialOurs
    maxSharpeFitnessPortfolioReturn = initialOurs
    NIFTY50Return = initialNifty
    print start
    columns = ['Buy Date', 'Sell Date', 'NIFTY50(Day)','NIFTY50(Cummulative)', 'Our Strategy(Day)','Our Strategy(Cummulative)']
    returnDf = pd.DataFrame(columns=columns)    
    while start<=end:
        start_ = start.strftime('%Y-%m-%d')
        next_ = (start + pd.DateOffset(days=1)).strftime('%Y-%m-%d')
        prev_ = (start - pd.DateOffset(days=1)).strftime('%Y-%m-%d')
        previousMonth_ = (start-pd.DateOffset(months=1)).strftime('%Y-%m-%d')

        rNIFTY = niftyReturnPercentage(start_,next_) #% 
        NIFTY50Return =  (1+(rNIFTY/100.0))*NIFTY50Return
        if(rNIFTY!=0):
            if(fitnessFunc == 'equalWeight'):
                wEqWeight = equalWeightVector(stocksList)
                wEqWeight = np.array(wEqWeight)/sum(wEqWeight)
                rEqWeight = portfolioReturnPercentage(wEqWeight, stocksList, start_, next_)
                equalWeightPortfolioReturn = (1+(rEqWeight/100.0))* equalWeightPortfolioReturn 
                values = [start,start + pd.DateOffset(days=1),rNIFTY,(NIFTY50Return-1)*100,rEqWeight,(equalWeightPortfolioReturn-1)*100]
            elif (fitnessFunc == 'marketCapitalizationWeight'):
                wMarCap = marketCapitalizationWeightVector(stocksList,start_)
                wMarCap = np.array(wMarCap)/sum(wMarCap)
                rMarCap = portfolioReturnPercentage(wMarCap, stocksList, start_, next_)
                marketCapitalizationPortfolioReturn = (1+(rMarCap/100.0))* marketCapitalizationPortfolioReturn
                values = [start,start + pd.DateOffset(days=1),rNIFTY,(NIFTY50Return-1)*100,rMarCap,(marketCapitalizationPortfolioReturn-1)*100]
        
            elif (fitnessFunc == 'minVarFitness'):
                wMinVar = geneticAlgorithm(totalPopulation,minVarFitness,stocksList,previousMonth_, prev_, generations)
                wMinVar = np.array(wMinVar)/sum(wMinVar)
                rMinVar = portfolioReturnPercentage(wMinVar, stocksList, start_, next_)
                minVarianceFitnessPortfolioReturn = (1+(rMinVar/100.0))* minVarianceFitnessPortfolioReturn
                values = [start,start + pd.DateOffset(days=1),rNIFTY,(NIFTY50Return-1)*100,rMinVar,(minVarianceFitnessPortfolioReturn-1)*100]
            elif (fitnessFunc == 'maxSharpeRatioFitness'):
                wMaxSharpe = geneticAlgorithm(totalPopulation,maxSharpeRatioFitness,stocksList,previousMonth_, prev_, generations)
                wMaxSharpe = np.array(wMaxSharpe)/sum(wMaxSharpe)
                rMaxSharpe = portfolioReturnPercentage(wMaxSharpe, stocksList, start_, next_)
                maxSharpeFitnessPortfolioReturn = (1+(rMaxSharpe/100.0))* maxSharpeFitnessPortfolioReturn
                values = [start,start + pd.DateOffset(days=1),rNIFTY,(NIFTY50Return-1)*100,rMaxSharpe,(maxSharpeFitnessPortfolioReturn-1)*100]
            print ('entry: ',values)
            returnDf = returnDf.append(pd.Series(values, index=returnDf.columns ), ignore_index=True)
        start = start + pd.DateOffset(days=1)
    if(fitnessFunc == 'equalWeight'):
        return returnDf, (NIFTY50Return), (equalWeightPortfolioReturn)
    elif (fitnessFunc == 'marketCapitalizationWeight'):
        return returnDf, (NIFTY50Return), (marketCapitalizationPortfolioReturn)
    elif (fitnessFunc == 'minVarFitness'):
        return returnDf, (NIFTY50Return), (minVarianceFitnessPortfolioReturn)
    elif (fitnessFunc == 'maxSharpeRatioFitness'):
        return returnDf, (NIFTY50Return), (maxSharpeFitnessPortfolioReturn)

#We want to do portfolio management between startDate and endDate. 
#Overall algorithm doh hyperparameter tuning followed by portfolio rebalancing.
#To decide hyper parameters (in tuning period), we select stocks using past one month return. To calculate past one month return on set of hyper parameters, we select set of stocks on weekly basis and rebalance it on daily basis.
#After tuning, we will use hyper parameters to do selection on weekly basis and rebalancing on daily basis.
def portfolioManagementAlgorithm(startDate, endDate):
    start = pd.to_datetime(np.datetime64(startDate, dtype='datetime64[D]'))
    end = pd.to_datetime(np.datetime64(endDate, dtype='datetime64[D]'))
    columns = ['Buy Date', 'Sell Date', 'NIFTY50(Day)','NIFTY50(Cummulative)', 'Our Strategy(Day)','Our Strategy(Cummulative)']
    returnDf = pd.DataFrame(columns=columns)
    initialNiftyReturns = 1
    initialOurReturns = 1
    while start<end:
        startTuning_ = (start - pd.DateOffset(months=1)).strftime('%Y-%m-%d')
        endTuning_ = (start - pd.DateOffset(days=1)).strftime('%Y-%m-%d')
        startTesting_ = start.strftime('%Y-%m-%d')
        endTesting_ = (start + pd.DateOffset(days=7)).strftime('%Y-%m-%d')        
        k,investorType,traingWinsowSize, maxReturnFunction = hyperParameterTuningInTrainingPeriod(startTuning_,endTuning_)
        startTrainingWindow = (start - pd.DateOffset(months=traingWinsowSize)).strftime('%Y-%m-%d')
        endTrainingWindow = (start - pd.DateOffset(days=1)).strftime('%Y-%m-%d')
        stocksList = StockSelection(startTrainingWindow,endTrainingWindow,k,investorType)
        df, niftyReturns, ourReturns = finalEvaluationInTestingPeriod(50,stocksList,startTesting_, endTesting_, 500,maxReturnFunction,initialNiftyReturns,initialOurReturns)
        initialNiftyReturns = niftyReturns
        initialOurReturns = ourReturns
        returnDf = returnDf.append(df)
        start = start + pd.DateOffset(days=7)
    return returnDf     

dfFirstQuarter = portfolioManagementAlgorithm('2017-01-01', '2017-04-01')

def plotFinalGraph(df):
    df["Buy Date"].dt.strftime('%d-%m-%Y')
    plt.clf()
    plt.style.use('seaborn-darkgrid')
    plt.plot('Buy Date', 'NIFTY50(Day)', data=df, marker='.', markerfacecolor='blue', markersize=6, color='skyblue', linewidth=1, label="Daily Nifty Return")
    plt.plot( 'Buy Date', 'NIFTY50(Cummulative)', data=df, marker='.', markerfacecolor='red', markersize=6, color='pink', linewidth=1, label="Overall Nifty Return")
    plt.plot( 'Buy Date', 'Our Strategy(Day)', data=df, marker='.', markerfacecolor='olive', markersize=6, color='green', linewidth=1,label="Our Daily Return")
    plt.plot( 'Buy Date', 'Our Strategy(Cummulative)', data=df, marker='.', markerfacecolor='black', markersize=6, color='yellow', linewidth=1, label="Our Overall Return")
    plt.legend(loc=2, ncol=1,fontsize=9)
    plt.xticks(fontsize=9, rotation=90)

    plt.title("Overall comparision of returns of our strategy with standard reference", loc='left', fontsize=12, fontweight=0, color='orange')
    plt.xlabel("Date")
    plt.ylabel("% Return")
    plt.show()
plotFinalGraph(dfFirstQuarter)

#Function to analyze different fitness(objective) functions for each of investor type.
def evaluateTestingPeriod(totalPopulation,stocksList,startDate, endDate, generations,initialNiftyReturns,initialEqualWeightPortfolioReturn, initialMarketCapitalizationPortfolioReturn,initialMinVarianceFitnessPortfolioReturn, initialMaxSharpeFitnessPortfolioReturn):
    start = pd.to_datetime(np.datetime64(startDate, dtype='datetime64[D]'))
    end = pd.to_datetime(np.datetime64(endDate, dtype='datetime64[D]'))
    equalWeightPortfolioReturn = initialEqualWeightPortfolioReturn
    marketCapitalizationPortfolioReturn = initialMarketCapitalizationPortfolioReturn
    minVarianceFitnessPortfolioReturn = initialMinVarianceFitnessPortfolioReturn
    maxSharpeFitnessPortfolioReturn = initialMaxSharpeFitnessPortfolioReturn
    NIFTY50Return = initialNiftyReturns
    columns = ['Buy Date', 'Sell Date', 'NIFTY50(Day)','NIFTY50(Cummulative)', 'Equal Weight(Day)','Equal Weight(Cummulative)', 'Market Cap(Day)','Market Cap(Cummulative)', 'Min Var(Day)','Min Var(Cummulative)', 'Max Sharpe(Day)','Max Sharpe(Cummulative)']
    returnDf = pd.DataFrame(columns=columns)    
    while start<=end:
        start_ = start.strftime('%Y-%m-%d')
        next_ = (start + pd.DateOffset(days=1)).strftime('%Y-%m-%d')
        prev_ = (start - pd.DateOffset(days=1)).strftime('%Y-%m-%d')
        previousMonth_ = (start-pd.DateOffset(months=1)).strftime('%Y-%m-%d')
        
        rNIFTY = niftyReturnPercentage(start_,next_) 
        NIFTY50Return =  (1+(rNIFTY/100.0))*NIFTY50Return
        
        wEqWeight = equalWeightVector(stocksList)
        wEqWeight = np.array(wEqWeight)/sum(wEqWeight)
        rEqWeight = portfolioReturnPercentage(wEqWeight, stocksList, start_, next_)
        equalWeightPortfolioReturn = (1+(rEqWeight/100.0))* equalWeightPortfolioReturn   
        
        if rEqWeight !=0:
            wMarCap = marketCapitalizationWeightVector(stocksList,start_)
            wMarCap = np.array(wMarCap)/sum(wMarCap)
            rMarCap = portfolioReturnPercentage(wMarCap, stocksList, start_, next_)
            marketCapitalizationPortfolioReturn = (1+(rMarCap/100.0))* marketCapitalizationPortfolioReturn
            
            wMinVar = geneticAlgorithm(totalPopulation,minVarFitness,stocksList,previousMonth_, prev_, generations)
            wMinVar = np.array(wMinVar)/sum(wMinVar)
            rMinVar = portfolioReturnPercentage(wMinVar, stocksList, start_, next_)
            minVarianceFitnessPortfolioReturn = (1+(rMinVar/100.0))* minVarianceFitnessPortfolioReturn

            wMaxSharpe = geneticAlgorithm(totalPopulation,maxSharpeRatioFitness,stocksList,previousMonth_, prev_, generations)
            wMaxSharpe = np.array(wMaxSharpe)/sum(wMaxSharpe)
            rMaxSharpe = portfolioReturnPercentage(wMaxSharpe, stocksList, start_, next_)
            maxSharpeFitnessPortfolioReturn = (1+(rMaxSharpe/100.0))* maxSharpeFitnessPortfolioReturn
        
            values = [start,start + pd.DateOffset(days=1),rNIFTY,(NIFTY50Return-1)*100,rEqWeight,(equalWeightPortfolioReturn-1)*100,rMarCap,(marketCapitalizationPortfolioReturn-1)*100,rMinVar,(minVarianceFitnessPortfolioReturn-1)*100,rMaxSharpe,(maxSharpeFitnessPortfolioReturn-1)*100]
            returnDf = returnDf.append(pd.Series(values, index=returnDf.columns ), ignore_index=True)
        start = start + pd.DateOffset(days=1)
    return returnDf, NIFTY50Return, equalWeightPortfolioReturn, marketCapitalizationPortfolioReturn, minVarianceFitnessPortfolioReturn, maxSharpeFitnessPortfolioReturn 

def compareFitnessFunctions(startDate, endDate, investorType):
    start = pd.to_datetime(np.datetime64(startDate, dtype='datetime64[D]'))
    end = pd.to_datetime(np.datetime64(endDate, dtype='datetime64[D]'))
    columns = ['Buy Date', 'Sell Date', 'NIFTY50(Day)','NIFTY50(Cummulative)', 'Equal Weight(Day)','Equal Weight(Cummulative)', 'Market Cap(Day)','Market Cap(Cummulative)', 'Min Var(Day)','Min Var(Cummulative)', 'Max Sharpe(Day)','Max Sharpe(Cummulative)']
    returnDf = pd.DataFrame(columns=columns)
    initialNiftyReturns = 1
    initialEqualWeightPortfolioReturn = 1
    initialMarketCapitalizationPortfolioReturn = 1
    initialMinVarianceFitnessPortfolioReturn = 1
    initialMaxSharpeFitnessPortfolioReturn = 1
    while start<end:
        startTesting_ = start.strftime('%Y-%m-%d')
        endTesting_ = (start + pd.DateOffset(days=7)).strftime('%Y-%m-%d')        
        startTrainingWindow = (start - pd.DateOffset(months=6)).strftime('%Y-%m-%d')
        endTrainingWindow = (start - pd.DateOffset(days=1)).strftime('%Y-%m-%d')
        stocksList = StockSelection(startTrainingWindow,endTrainingWindow,5,investorType)
        df, niftyReturns, equalWeightPortfolioReturn, marketCapitalizationPortfolioReturn, minVarianceFitnessPortfolioReturn, maxSharpeFitnessPortfolioReturn = evaluateTestingPeriod(50,stocksList,startTesting_, endTesting_, 500,initialNiftyReturns,initialEqualWeightPortfolioReturn, initialMarketCapitalizationPortfolioReturn,initialMinVarianceFitnessPortfolioReturn, initialMaxSharpeFitnessPortfolioReturn)
        initialNiftyReturns = niftyReturns
        initialEqualWeightPortfolioReturn = equalWeightPortfolioReturn
        initialMarketCapitalizationPortfolioReturn = marketCapitalizationPortfolioReturn
        initialMinVarianceFitnessPortfolioReturn = minVarianceFitnessPortfolioReturn
        initialMaxSharpeFitnessPortfolioReturn = maxSharpeFitnessPortfolioReturn
        returnDf = returnDf.append(df)
        start = start + pd.DateOffset(days=7)
    return returnDf     

foreignDf = compareFitnessFunctions('2017-01-01', '2017-04-01', 'Foreign')

#plot the graph of return by each of objective function to get weight vector for given investor type.
def plotTheGraph(df,investorType):
    df["Buy Date"].dt.strftime('%d-%m-%Y')
    plt.clf()
    plt.style.use('seaborn-darkgrid')

    plt.plot('Buy Date', 'Equal Weight(Cummulative)', data=df, marker='.', markerfacecolor='blue', markersize=6, color='skyblue', linewidth=1, label="EW")
    plt.plot( 'Buy Date', 'Market Cap(Cummulative)', data=df, marker='.', markerfacecolor='red', markersize=6, color='pink', linewidth=1, label="MC")
    plt.plot( 'Buy Date', 'Min Var(Cummulative)', data=df, marker='.', markerfacecolor='olive', markersize=6, color='green', linewidth=1,label="Var")
    plt.plot( 'Buy Date', 'Max Sharpe(Cummulative)', data=df, marker='.', markerfacecolor='black', markersize=6, color='yellow', linewidth=1, label="Sharpe")
    plt.plot( 'Buy Date', 'NIFTY50(Cummulative)', data=df, marker='.', markerfacecolor='Brown', markersize=6,color='orange', linewidth=1, label="NIFTY50")

    plt.legend(loc=2, ncol=2,fontsize=9)
    plt.xticks(fontsize=9, rotation=90)

    plt.title("Comparision of returns by different fitness functions for "+ investorType +" investors", loc='left', fontsize=12, fontweight=0, color='orange')
    plt.xlabel("Date")
    plt.ylabel("% Return")
    plt.show()
plotTheGraph(foreignDf,'foreign')