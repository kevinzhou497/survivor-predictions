import pyreadr
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm 
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor


# retrieving vote history as a panda dataframe
vote_history = pyreadr.read_r('vote_history.rda')
voteHistory = vote_history["vote_history"]

# only keep the US seasons
voteHistory = voteHistory.loc[voteHistory['version'] == 'US']
# voteHistory.to_csv("vote_history.csv")
lastSeason = int(voteHistory["season"].max())

juryVotes = pyreadr.read_r('jury_votes.rda')
juryVotes = juryVotes["jury_votes"]
juryVotes = juryVotes.loc[juryVotes['version'] == 'US']
juryVotes.to_csv("jury_votes.csv")
finalists = juryVotes.finalist.unique()
# group them by season
seasonFinalists = {}
# get the seasons
finalistTrack = 0


for i in range(1, 44):
    currSeasonJuryVotes = juryVotes.loc[juryVotes["season"] == i]
    currFinalists = currSeasonJuryVotes.finalist.unique()
    seasonFinalists[i] = currFinalists
    
'''for i in range(1, 13):
    currentFinalists = []
    #print(finalists[finalistTrack])
    currentFinalists.append(finalists[finalistTrack])
    currentFinalists.append(finalists[finalistTrack+1])
    seasonFinalists[i] = currentFinalists
    finalistTrack += 2
finalistTrack = 24

for i in range(13, 44):
    # we want the 25th finalist
    
 # season 13 onward is when there was 3 finalists (not always)
 # just get the unique valeus when its that season, and add those to the list
    currentFinalists = []
    if finalistTrack < 102:
        currentFinalists.append(finalists[finalistTrack])
    else:
        break
    currentFinalists.append(finalists[finalistTrack+1])
    currentFinalists.append(finalists[finalistTrack+2])
    seasonFinalists[i] = currentFinalists
    finalistTrack += 3'''
#print(seasonFinalists)


 

# have a list of dataframes corresponding to each season
seasons = []

# retrieving winners in order of season
seasonSummary = pyreadr.read_r('season_summary.rda')
seasonSummary = seasonSummary["season_summary"]
seasonSummary = seasonSummary.loc[seasonSummary['version'] == 'US']
winners = seasonSummary["winner"].values.tolist()

#print(winners)


# have a loop that goes through seasons and appends to the list of dataframes
for i in range(1, 44):
    seasonNum = float(i)
    seasonVotes = voteHistory.loc[voteHistory["season"] == seasonNum]
    seasons.append(seasonVotes)
    
# make a matrix (2-D array) for each contestant for each episode of each season
# structure:
# {season : {episode : {contestant : matrix}}}
networks = {}
# aggregate results over an episode per contestant
aggNet = {}
centralities = {}

seasonNum = 0
contestantDicts = []
trainingDict = {"contestant": [], "season": [], "degree": [], "eigenvector" :[], "katz": [], "closeness": [], "betweenness":[], "winner": []}
testingDict = {"contestant": [], "season": [], "degree": [], "eigenvector" :[], "katz": [], "closeness": [], "betweenness":[], "winner": []}
fullDict = {"contestant": [], "season": [], "degree": [], "eigenvector" :[], "katz": [], "closeness": [], "betweenness":[], "winner": []}
numEps = []
# dictionary with key: candidate and value: episode of elimination
candidatesElim = {}
for season in seasons:
    
    seasonNum += 1
    # list of contestants, have a dictionary so can assign them numbers
    contestants = season.castaway.unique()
    numContestants = len(contestants)
    
    # should maintain this info somewhere
    contestantDict = {}
    contestNumber = 0
    for contestant in contestants:
        contestantDict[contestant] = contestNumber
        contestNumber+= 1
    contestantDicts.append(contestantDict)
        
    lastEpisode = int(season["episode"].max())
    numEps.append(lastEpisode)
    
    # go through each episode
    seasonName = "season" + str(seasonNum)
    networks[seasonName] = {}
    aggNet[seasonName] = {}
    aggSeason = []
    candidatesElim[seasonName] = {}
    for i in range(numContestants):
                aggSeason.append([0. for i in range(numContestants)])
    aggSeason = np.asarray(aggSeason)

    # now we need to keep track of when contestants are eliminated
    for episode in range(1, lastEpisode+1):
        episodeVotes = season.loc[season["episode"] == float(episode)]
        #epCandidates = episodeVotes.castaway.unique()
        votedOut = episodeVotes["voted_out"]
        for cand in votedOut:
            candidatesElim[seasonName][cand] = episode
            
        # shouldnt reset it everytime maybe? only happening on the last contestant 
        networks[seasonName][episode] = {}
        aggNet[seasonName][episode] = {}
        # print(episodeVotes)
        aggEpisode = []
        for i in range(numContestants):
                aggEpisode.append([0 for i in range(numContestants)])
        aggEpisode = np.asarray(aggEpisode)
                
        for contestant in contestants:
            # make a matrix for this episode for each contestant
            contestantNet = []
            for i in range(numContestants):
                contestantNet.append([0 for i in range(numContestants)])
                
            # make np array to make it easier to work with networkx
            contestantNet = np.asarray(contestantNet)
            
        
            # everyone in this list should be marked 1 with each other in the matrix
            # everyone who voted for this person
            votedFor = episodeVotes.loc[episodeVotes["vote"] == contestant]["castaway"].unique()
            # print(votedFor)
            #print(votedFor)
            
            # to tell when someone is eliminated, can look at the list of voters for that episode
            # have a dictionary that is the episode that someone is eliminated 
            # plot degree vs episode that someone is eliminated
            for person in votedFor:
                for person2 in votedFor:
                    contestantNet[contestantDict[person]][contestantDict[person2]] = 1
                    # maybe doing this too much? need to have it for each person, aggregated wouldnt have it for each person
            aggEpisode += contestantNet
      
            G = nx.from_numpy_array(contestantNet)
            
            
            # tuple with adjacency matrix and the networkx graph
            networks[seasonName][episode][contestant] = (contestantNet)
            
        
        # aggregate for each episode
        # this one has for each episode, aggregated over the contestants
        # divide by row sum 
        #print(aggEpisode)
        #print(aggEpisode.shape[0])
        # dont loop through it like this
        aggEpisode = aggEpisode/aggEpisode.sum(axis=1, keepdims=True)
        aggEpisode = np.nan_to_num(aggEpisode)
        # replace NaNs with 0's 
        #print(aggEpisode)

        '''for row in range(aggEpisode.shape[0]):
            #print(np.sum(aggEpisode[row]))
           
            #print(aggEpisode[row])
            currSum = np.sum(aggEpisode[row])
            #aggEpisode[row] = aggEpisode[row] / float(currSum)
            for i in range(len(aggEpisode[row])):
                if currSum != 0:
                    currVal = aggEpisode[row][i]    
                    
                    print("1")      
                    print(type(currVal))
                    print("2")
                    print(type(currSum))
                    print("3")
                    print(float(currVal))
                    print(float(currSum))  
                    print(float(currVal) / float(currSum))
                          
                    aggEpisode[row][i] = (float(currVal) / float(currSum))
                    
                    #print(aggEpisode[row][i])
                    #print(currSum)
                #aggEpisode[row]
        print(aggEpisode)'''
            #aggEpisode[row] = (aggEpisode[row]) / float((np.sum(aggEpisode[row])))
          
            #print(aggEpisode[row])
            
    
        aggNet[seasonName][episode] = aggEpisode
        # for some reason its not doing decimals properly
        #print(aggEpisode)
        
        # make one for each contestant, aggregated over the episodes
        aggSeason += aggEpisode
    
    aggNet[seasonName] = (aggSeason)
    #print(aggSeason)
    # get the centralities here 
    # Degree, eigenvector, 
    # katz (alpha values of 0.2, 0.8), closeness centrality,
    # betweeness centrality, vote rank
    centralities[seasonName] = {}
    G = nx.from_numpy_array(aggNet[seasonName])
    centralities[seasonName]["degree"] = nx.degree_centrality(G)
    centralities[seasonName]["eigenvector"] = nx.eigenvector_centrality(G, tol=1.0e-3)
    
    # 'power iteration failed to converge within 1000 iterations')
    # make katz 1/2 eigen
    
    L = nx.normalized_laplacian_matrix(G)
    e = np.linalg.eigvals(L.toarray())
    maxEigen = max(e)
    #print(maxEigen)
    centralities[seasonName]["katz"] = nx.katz_centrality_numpy(G, alpha = 0.1, weight= 'weight')
    # set the alpha to 1/ 2 * lambda max (eigenvalue)
    
    # centralities[seasonName]["katz8"] = nx.katz_centrality(G, 0.9, max_iter = 100000)
    #numpy.linalg.LinAlgError: Singular matrix
    centralities[seasonName]["closeness"] = nx.closeness_centrality(G)
    centralities[seasonName]["betweenness"] = nx.betweenness_centrality(G)
    voteranked = nx.voterank(G)
    #print(voteranked)
   # centralities[seasonName]["voterank"] = nx.voterank(G)
   # highest gets 1, 2nd place gets (n-1) / n, 
   # see what context is used, if its relevant
    
    # also need to keep track of season to mark contestants

    fullDict["contestant"].extend(contestants)
    fullDict["degree"].extend(centralities[seasonName]["degree"].values())
    fullDict["eigenvector"].extend(centralities[seasonName]["eigenvector"].values())
    fullDict["katz"].extend(centralities[seasonName]["katz"].values())
    fullDict["closeness"].extend(centralities[seasonName]["closeness"].values())
    fullDict["betweenness"].extend(centralities[seasonName]["betweenness"].values())
    winMarker = []
    currSeasonNum = []
    for contestant in contestants:
        currSeasonNum.append(seasonNum)
        if contestant == winners[seasonNum-1]:
            winMarker.append(1)
        else:
            winMarker.append(0)
    #print(currSeasonNum)
    fullDict["winner"].extend(winMarker)
    fullDict["season"].extend(currSeasonNum)
# each candidate has data corresponding to it, for odd seasons all these rows can be in the same data frame (centralities and then winner)
    if seasonNum % 2 == 1:
        # make a df row with the centralities and then if the person won
        # loop through the contestants in each season
        trainingDict["contestant"].extend(contestants)
        trainingDict["degree"].extend(centralities[seasonName]["degree"].values())
        trainingDict["eigenvector"].extend(centralities[seasonName]["eigenvector"].values())
        trainingDict["katz"].extend(centralities[seasonName]["katz"].values())
        trainingDict["closeness"].extend(centralities[seasonName]["closeness"].values())
        trainingDict["betweenness"].extend(centralities[seasonName]["betweenness"].values())
        
       # trainingDict["voterank"].extend(centralities[seasonName]["voterank"].values())
        # make winners list
        winMarker = []
        currSeasonNum = []
        for contestant in contestants:
            currSeasonNum.append(seasonNum)
            if contestant == winners[seasonNum-1]:
                winMarker.append(1)
            else:
                winMarker.append(0)
            
        trainingDict["winner"].extend(winMarker)
        trainingDict["season"].extend(currSeasonNum)
    else:
        testingDict["contestant"].extend(contestants)
        testingDict["degree"].extend(centralities[seasonName]["degree"].values())
        testingDict["eigenvector"].extend(centralities[seasonName]["eigenvector"].values())
        testingDict["katz"].extend(centralities[seasonName]["katz"].values())
        testingDict["closeness"].extend(centralities[seasonName]["closeness"].values())
        testingDict["betweenness"].extend(centralities[seasonName]["betweenness"].values())
      
        # how to do vote rank with the other features?
       # trainingDict["voterank"].extend(centralities[seasonName]["voterank"].values())
        # make winners list
        winMarker = []
        currSeasonNum = []
        for contestant in contestants:
            currSeasonNum.append(seasonNum)
            
            if contestant == winners[seasonNum-1]:
                winMarker.append(1)
            else:
                winMarker.append(0)
            
        testingDict["winner"].extend(winMarker)
        testingDict["season"].extend(currSeasonNum)

print(candidatesElim)    
# print(trainingDict)
#print(contestantDict)
#print(aggNet["season43"])
#print(trainingDict)
# train the model 
# print(centralities)
# print(trainingDict)
#print(aggNet)
trainDf = pd.DataFrame(trainingDict)
#print(trainDf)
testDf = pd.DataFrame(testingDict)
fullDf = pd.DataFrame(fullDict)

# training with statsmodels
model = smf.ols(formula='winner ~ degree + eigenvector + katz + closeness + betweenness', data = trainDf )
res = model.fit()
print(res.summary())
# now train the model with SK learn
x_train = trainDf[["degree", "eigenvector", "katz", "closeness", "betweenness"]]
#x_train = trainDf[["degree"]]
x_trainE = trainDf[["eigenvector"]]
y_train = trainDf["winner"]
LR = LinearRegression()
LR.fit(x_train, y_train)

x_test = testDf[["degree", "eigenvector", "katz", "closeness", "betweenness"]]
x_testE = testDf[["eigenvector"]]
y_test = testDf["winner"]
y_pred = LR.predict(x_test)
score = r2_score(y_test, y_pred)

regressor = RandomForestRegressor(n_estimators=100, random_state = 0)
regressor.fit(x_train, y_train)
# coefficients of regression
#print(LR.coef_)
print(1/3)

# get the finalists for each season, get the centralities and predict
# take the highest probability contestant and see if it equals the actual winner

probabilities = {}
allProbs = {}

# loop through the finalists
verifying = []
verifyingFinalists = []
for season in seasonFinalists:
    # check the probabilities of all of them
    probabilities[season] = {}
    finalists = seasonFinalists[season]
    #print(finalists)
    # finalists = pd.DataFrame(finalists)
    
    # ne
    # need to get the centrality measures and such
    ''' trainingDict["contestant"].extend(contestants)
    trainingDict["degree"].extend(centralities[seasonName]["degree"].values())
    trainingDict["eigenvector"].extend(centralities[seasonName]["eigenvector"].values())
    trainingDict["katz"].extend(centralities[seasonName]["katz"].values())
    trainingDict["closeness"].extend(centralities[seasonName]["closeness"].values())
    trainingDict["betweenness"].extend(centralities[seasonName]["betweenness"].values())'''
    maxProb = 0
    maxCont = ""
    maxContAll = ""
    # also want to see probability of making it to the finalists
    # get the probabilities of all contestants in a season
    # see if the highest probability contestant made it to the finals
    # another option is training on finalist as the y - ask about this 
    
    checkingFinalists = fullDf.loc[fullDf['season'] == season]
    maxPred = 0
    for contestant in checkingFinalists['contestant']:
        checkingFinalist = checkingFinalists.loc[checkingFinalists['contestant'] == contestant]
        #currPrediction = res.predict(checkingFinalist[["degree", "eigenvector", "katz", "closeness", "betweenness"]])
        currPrediction = regressor.predict(checkingFinalist[["degree", "eigenvector", "katz", "closeness", "betweenness"]])
        #currPrediction.value[0] with statsmodels
        if currPrediction > maxPred:
            maxPred = currPrediction
            maxContAll = contestant
    if maxContAll in finalists:
        verifyingFinalists.append(1)
    else:
        verifyingFinalists.append(0)
        
    for finalist in finalists:
        
        #currDict = {"contestant": [], "degree": [], "eigenvector" :[], "katz": [], "closeness": [], "betweenness":[], "winner": []}
        #print(fullDf)
        currData = fullDf.loc[fullDf['season'] == season]
        # the finalist thing seems to return no data
        #print(currData)
       
        currData = currData.loc[currData['contestant'] == finalist]
        
        
        '''currDict["contestant"] = currData["contestant"]
        degree = currData["degree"]
        eigen = currData["eigenvector"]
        katz = currData["katz"]
        closeness = currData["closeness"]
        betweenness = currData["betweenness"]'''
        #probability = LR.predict(currData[["degree", "eigenvector", "katz", "closeness", "betweenness"]])
        #probability = res.predict(currData[["degree", "eigenvector", "katz", "closeness", "betweenness"]])
        probability = regressor.predict(currData[["degree", "eigenvector", "katz", "closeness", "betweenness"]])

        #probability = LR.predict(currData[["degree"]])
        #print(probability)
        #currSeasonName = "season" + season
        #eigen = centralities[currSeasonName][]
        probabilities[season][finalist] = probability
        if probability > maxProb:
            maxProb = probability
            maxCont = finalist
    if maxCont == winners[season-1]:
        verifying.append(1)
    else:
        verifying.append(0)
        
        
    # select the key with the largest probability, see if it matches the winner
    

print("winners predictions") 
print(verifying)
print("finalists predictions")
print(verifyingFinalists)

        
#print(seasonFinalists)
pickle.dump(seasonFinalists, open("finalists.pkl", "wb"))
pickle.dump(contestantDicts, open("contestant_mappings.pkl", "wb"))
#print(contestantDicts)


# plotting the degree vs elimination episode (actually episode / (total episodes+1)) using matplotlib
# https://matplotlib.org/stable/gallery/shapes_and_collections/scatter.html
# need to turn the contestant into a number so it matches with the degrees output
# should both be a list where the indices match up
for i in range(1,44):
    seasonName = "season" + str(i)
    seasonCandElims = candidatesElim[seasonName]
    degrees = centralities[seasonName]["degree"]
    eigen = centralities[seasonName]["eigenvector"]
    # go through the candidates and get their elimination 
    candElimsList = []
    # might just be same order as seasonCandElims
    print(contestantDicts[i-1])
    print(seasonCandElims)
    currNumEps = numEps[i-1]
    
    for cont in contestantDicts[i-1]:
        # this will give the elimEpisode for 
        if cont in seasonCandElims:
            elimEp = seasonCandElims[cont] / (currNumEps+1)
            # divide over total eps + 1
            candElimsList.append(elimEp)
        else:
            elimEp = (currNumEps) / (currNumEps+1)
            candElimsList.append(elimEp)
    # need degrees to be a list, also need to factor in finalists
    plt.scatter(eigen.values(), candElimsList)
    #plt.show()
counter = 0
for i in range(len(verifying)):
    if verifying[i] == 1:
        counter += 1
print(counter)
# 12 / 2 + 31 / 3    
'''
    
    
N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()'''
        




# get a predictor function with a mix of other characteristics too?
        
        

    

# there are centralities for each contestant

# make a dataframe, with columns for the centralities (will be X) and then the winner (will be Y)

                    
# print(networks)

# use the centralities in a multiple linear regression and get the coefficients
    

# just call these A, store the data in somewhere so I can send it over in a file (also send over the winners)
# print(aggNet["season43"])

# store the data in pkl files 
# want both the network matrix and the winners for each season
for i in range(1, 44):
    szn = "season" + str(i)
    pklName = szn + "New.pkl"
    winnerPkl = szn + "winner.pkl"
    pickle.dump(aggNet[szn], open(pklName, "wb"))
    # want the number of the winners
    pickle.dump(contestantDicts[i-1][winners[i-1]], open(winnerPkl, "wb"))
    

A1 = pickle.load(open("season1.pkl", "rb"))
W1 = pickle.load(open("season1winner.pkl", "rb"))

#print(contestantDicts[8])
# just give the number for the winner
# send over a zip files
# print(W1)
# print(A1)


# what type of draw to use? doesnt matter for now
# nx.draw(aggNet["season43"][1])
# plt.show()
