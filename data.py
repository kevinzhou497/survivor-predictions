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
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from xgboost import XGBRegressor


two_finalist_seasons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 18]
# retrieving vote history as a panda dataframe
vote_history = pyreadr.read_r('vote_history.rda')
voteHistory = vote_history["vote_history"]

castaways = pyreadr.read_r('castaways.rda')
castaways = castaways["castaways"]
castaways = castaways.loc[castaways["version"] == 'US']

castawayDetails = pyreadr.read_r('castaway_details.rda')
castawayDetails = castawayDetails["castaway_details"]
castawayDetails.to_csv("castaway_details.csv")

challenge_results = pyreadr.read_r('challenge_results.rda')
challengeResults = challenge_results["challenge_results"]

challengeResults = challengeResults.loc[challengeResults["version"] == 'US']
#challengeResults = challengeResults.loc[challengeResults['tribe_status'] == 'Merged']
challengeResults.to_csv("challenge_results.csv")

challengeWins = {}
# only keep the US seasons
voteHistory = voteHistory.loc[voteHistory['version'] == 'US']
# voteHistory.to_csv("vote_history.csv")
lastSeason = int(voteHistory["season"].max())

# only have it when tribe status is merged -> leads to better results
voteHistory = voteHistory.loc[voteHistory['tribe_status'] == 'Merged']

juryVotes = pyreadr.read_r('jury_votes.rda')
juryVotes = juryVotes["jury_votes"]
juryVotes = juryVotes.loc[juryVotes['version'] == 'US']
juryVotes.to_csv("jury_votes.csv")

finalists = juryVotes.finalist_id.unique()
# group them by season
seasonFinalists = {}
# get the seasons
finalistTrack = 0

castaways.to_csv("castaways.csv")
castaways = castaways.drop(labels=[496, 499, 767, 775, 788, 796, 875, 870, 1085, 1088, 1120, 1129], axis=0)
#castaways = castaways.drop_duplicates(subset=['castaway_id'])

# handle the different regions seasons separately with the preprocessing

# first add gender as a feature

for i in range(1, 44):
    currSeasonJuryVotes = juryVotes.loc[juryVotes["season"] == i]
    currFinalists = currSeasonJuryVotes.finalist_id.unique()
    seasonFinalists[i] = currFinalists
    

# have a list of dataframes corresponding to each season
seasons = []

# retrieving winners in order of season
seasonSummary = pyreadr.read_r('season_summary.rda')
seasonSummary = seasonSummary["season_summary"]
seasonSummary = seasonSummary.loc[seasonSummary['version'] == 'US']
seasonSummary.to_csv("season_summary.csv")
winners = seasonSummary["winner_id"].values.tolist()

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
directedNetworks = {}

seasonNum = 0
contestantDicts = []
# add in CON centrality from the Bonato paper 
# CON definition: Common out neighbors
trainingDict = {"contestant": [], "season": [], "age": [], "gender": [], "challenges_won" : [], "degree": [], "eigenvector" :[], "katz": [], "closeness": [], "betweenness":[], "in_degree":[], "con_score":[], "winner": [], "finalist": []}
testingDict = {"contestant": [], "season": [], "age": [], "gender" : [], "challenges_won":[], "degree": [], "eigenvector" :[], "katz": [], "closeness": [], "betweenness":[], "in_degree": [], "con_score":[], "winner": [], "finalist": []}
fullDict = {"contestant": [], "season": [], "age": [], "gender" : [], "challenges_won" : [], "degree": [], "eigenvector" :[], "katz": [], "closeness": [], "betweenness":[], "in_degree": [], "con_score" : [], "winner": [], "finalist": []}
numEps = [] 


# looping through challenge results 
for index in challengeResults.index:
    currSeason = int(challengeResults["season"][index])
    challengeWins[currSeason] = {}
    if challengeResults["result"][index] == "won":
        if challengeResults["castaway"][index] in challengeWins[currSeason]:
            challengeWins[currSeason][challengeResults["castaway_id"][index]] += 1
        else:
            challengeWins[currSeason][challengeResults["castaway_id"][index]] = 1 

# dictionary with key: candidate and value: episode of elimination
candidatesElim = {}
for season in seasons:
    
    seasonNum += 1
    castawaySeason = castaways.loc[castaways["season"] == seasonNum]
    
    # list of contestants, have a dictionary so can assign them numbers
    contestants = season.castaway_id.unique()
    numContestants = len(contestants)
    
    # should maintain this info somewhere
    contestantDict = {}
    contestNumber = 0
    # these are all after the merge
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
    directedNetworks[seasonName] = {}
    aggSeason = []
    directedSeason = []
    candidatesElim[seasonName] = {}
    
    # doing this for original matrices
    for i in range(numContestants):
                aggSeason.append([0. for i in range(numContestants)])
    aggSeason = np.asarray(aggSeason)
    
    # doing it for directed matrices
    for i in range(numContestants):
                directedSeason.append([0. for i in range(numContestants)])
    directedSeason = np.asarray(directedSeason)
    

    # now we need to keep track of when contestants are eliminated
    for episode in range(1, lastEpisode+1):
        episodeVotes = season.loc[season["episode"] == float(episode)]
        #print(episodeVotes)
        #epCandidates = episodeVotes.castaway.unique()
        votedOut = episodeVotes["voted_out_id"]
        for cand in votedOut:
            candidatesElim[seasonName][cand] = episode
            
        # shouldnt reset it everytime maybe? only happening on the last contestant 
        networks[seasonName][episode] = {}
        aggNet[seasonName][episode] = {}
        
        # print(episodeVotes)
        
        
        # doing it for original matrices
        aggEpisode = []
        for i in range(numContestants):
                aggEpisode.append([0 for i in range(numContestants)])
        aggEpisode = np.asarray(aggEpisode)
        
        # doing it for directed matrices
        directedEpisode = []
        for i in range(numContestants):
                directedEpisode.append([0 for i in range(numContestants)])
        directedEpisode = np.asarray(directedEpisode)
        
        # do we want to loop through contestants
                
        for contestant in contestants:
            # make a matrix for this episode for each contestant
            contestantNet = []
            for i in range(numContestants):
                contestantNet.append([0 for i in range(numContestants)])
                
            # make np array to make it easier to work with networkx
            contestantNet = np.asarray(contestantNet)
            
            # everyone in this list should be marked 1 with each other in the matrix
            # everyone who voted for this person
            votedFor = episodeVotes.loc[episodeVotes["vote_id"] == contestant]["castaway_id"].unique()
            # to tell when someone is eliminated, can look at the list of voters for that episode
            # have a dictionary that is the episode that someone is eliminated 
            # plot degree vs episode that someone is eliminated
            for person in votedFor:
                # directed network
                directedEpisode[contestantDict[person]][contestantDict[contestant]] += 1
                
                # original network
                for person2 in votedFor:
                    if person == person2:
                        continue
                    contestantNet[contestantDict[person]][contestantDict[person2]] = 1
                    # maybe doing this too much? need to have it for each person, aggregated wouldnt have it for each person
            aggEpisode += contestantNet
      
            G = nx.from_numpy_array(contestantNet)
            
            
            # tuple with adjacency matrix and the networkx graph
            networks[seasonName][episode][contestant] = (contestantNet)
            
        '''aggEpisode = aggEpisode/aggEpisode.sum(axis=1, keepdims=True)
        aggEpisode = np.nan_to_num(aggEpisode)'''
        
        aggNet[seasonName][episode] = aggEpisode
        directedNetworks[seasonName][episode] = directedEpisode
        # for some reason its not doing decimals properly
        #print(aggEpisode)
        
        # make one for each contestant, aggregated over the episodes
        aggSeason += aggEpisode
        
        directedSeason += directedEpisode
    
    aggNet[seasonName] = (aggSeason)
    directedNetworks[seasonName] = directedSeason
    #print(aggSeason)
    # get the centralities here 
    # Degree, eigenvector, 
    # katz (alpha values of 0.2, 0.8), closeness centrality,
    # betweeness centrality, vote rank
    centralities[seasonName] = {}
    G = nx.from_numpy_array(aggNet[seasonName])
    centralities[seasonName]["degree"] = nx.degree_centrality(G)
    centralities[seasonName]["eigenvector"] = nx.eigenvector_centrality(G)
    
    # 'power iteration failed to converge within 1000 iterations')
    # make katz 1/2 eigen
    
    L = nx.normalized_laplacian_matrix(G)
    e = np.linalg.eigvals(L.toarray())
    maxEigen = max(e)
    #print(maxEigen)
    centralities[seasonName]["katz"] = nx.katz_centrality_numpy(G, alpha = .1, weight= 'weight')
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
    
    seasonWinner = seasonSummary.loc[seasonSummary["season"]== float(seasonNum)]
    winner = seasonWinner["winner"].item()
    # winner + finalist markers
    winMarker = []
    currSeasonNum = []
    finalists = seasonFinalists[seasonNum]
    finalistMarker = []
    ages = []
    genders = []
    challengeWinsList = []
    inDegrees = []
    conScores = []
    for contestant in contestants:
        if contestant in challengeWins[seasonNum]:
            challengeWinsList.append(challengeWins[seasonNum][contestant])
        else:
            challengeWinsList.append(0)
        castawaySeasonCont = castawaySeason.loc[castawaySeason["castaway_id"] == contestant]
        #print(castawaySeasonCont["age"])
        #print(contestant)
        currCastaway = castawayDetails.loc[castawayDetails['castaway_id'] == contestant]
        gender = currCastaway["gender"].item()
        if (gender == "Male"):
            genders.append(0)
        else:
            genders.append(1)
        
        age = castawaySeasonCont["age"].item()
        
        assert (type(age) is float)
        ages.append(age)
        
        
        currSeasonNum.append(seasonNum)
        if contestant == winners[seasonNum-1]:
            winMarker.append(1)
        else:
            winMarker.append(0)
        if contestant in finalists:
            finalistMarker.append(1)
        else:
            finalistMarker.append(0)
        
        
        # get directed networks data here
        contestNum = contestantDict[contestant]
        inDegreeSum = (directedNetworks[seasonName])[:, contestNum].sum()
        inDegrees.append(inDegreeSum)
         
        # calculating CON 
        # use directedNetworks[seasonName]
        
    for i in range(len(directedNetworks[seasonName])):
        runSum = 0 
        # still need to go through the rows 
        for j in range(len(directedNetworks[seasonName])):
            if i == j:
                continue 
            
            for k in range(len(directedNetworks[seasonName][0])):
                runSum += min(directedNetworks[seasonName][i][k], directedNetworks[seasonName][j][k] )
            
        
        conScores.append(runSum)
            
            
            
        
        # iterate through the rows 
        
        
        '''if contestant in candidatesElim[seasonName]:
            episodeElim = candidatesElim[seasonName][contestant]
            elimScore = episodeElim / (lastEpisode+1)
        elif contestant in finalists:
            elimScore = lastEpisode / (lastEpisode+1)
        else:
            elimScore = 5
            
        winMarker.append(elimScore)'''
            
    fullDict["winner"].extend(winMarker)
    fullDict["season"].extend(currSeasonNum)
    fullDict["finalist"].extend(finalistMarker)
    fullDict["age"].extend(ages)
    fullDict["gender"].extend(genders)
    fullDict["challenges_won"].extend(challengeWinsList)
    fullDict["in_degree"].extend(inDegrees)
    fullDict["con_score"].extend(conScores)
    
# training is used on odd seasons
# each candidate has data corresponding to it, for odd seasons all these rows can be in the same data frame (centralities and then winner)
    if seasonNum % 2 == 0:
        # make a df row with the centralities and then if the person won
        # loop through the contestants in each season 
        # can get the other features here too 
        trainingDict["contestant"].extend(contestants)
        trainingDict["degree"].extend(centralities[seasonName]["degree"].values())
        trainingDict["eigenvector"].extend(centralities[seasonName]["eigenvector"].values())
        trainingDict["katz"].extend(centralities[seasonName]["katz"].values())
        trainingDict["closeness"].extend(centralities[seasonName]["closeness"].values())
        trainingDict["betweenness"].extend(centralities[seasonName]["betweenness"].values())
    
        # make winners list
        winMarker = []
        currSeasonNum = [] 
        finalists = seasonFinalists[seasonNum]
        finalistMarker = []
        ages = []
        genders = []
        challengeWinsList = []
        inDegrees = []
        conScores = []
        
        for contestant in contestants:
            if contestant in challengeWins[seasonNum]:
                challengeWinsList.append(challengeWins[seasonNum][contestant])
            else:
                challengeWinsList.append(0)
            currCastaway = castawayDetails.loc[castawayDetails['castaway_id'] == contestant]
            gender = currCastaway["gender"].item()
            if (gender == "Male"):
                genders.append(0)
            else:
                genders.append(1)
            castawaySeasonCont = castawaySeason.loc[castawaySeason["castaway_id"] == contestant]
            age = castawaySeasonCont["age"].item()
            ages.append(age)
            currSeasonNum.append(seasonNum)
            if contestant == winners[seasonNum-1]:
                winMarker.append(1)
            else:
                winMarker.append(0)
            if contestant in finalists:
                finalistMarker.append(1)
            else:
                finalistMarker.append(0)
                '''
            if contestant in candidatesElim[seasonName]:
                episodeElim = candidatesElim[seasonName][contestant]
                elimScore = episodeElim / (lastEpisode+1)
            elif contestant in finalists:
                elimScore = lastEpisode / (lastEpisode+1)
            else:
                elimScore = 5
            winMarker.append(elimScore)'''
            contestNum = contestantDict[contestant]
            inDegreeSum = (directedNetworks[seasonName])[:, contestNum].sum()
            inDegrees.append(inDegreeSum)
            
        for i in range(len(directedNetworks[seasonName])):
            runSum = 0 
        # still need to go through the rows 
            for j in range(len(directedNetworks[seasonName])):
                if i == j:
                    continue 
                
                for k in range(len(directedNetworks[seasonName][0])):
                    runSum += min(directedNetworks[seasonName][i][k], directedNetworks[seasonName][j][k] )
            
        
            conScores.append(runSum)
            
        trainingDict["winner"].extend(winMarker)
        trainingDict["season"].extend(currSeasonNum)
        trainingDict["finalist"].extend(finalistMarker)
        trainingDict["age"].extend(ages)
        trainingDict["gender"].extend(genders)
        trainingDict["challenges_won"].extend(challengeWinsList)
        trainingDict["in_degree"].extend(inDegrees)
        trainingDict["con_score"].extend(conScores)
    else:
        testingDict["contestant"].extend(contestants)
        testingDict["degree"].extend(centralities[seasonName]["degree"].values())
        testingDict["eigenvector"].extend(centralities[seasonName]["eigenvector"].values())
        testingDict["katz"].extend(centralities[seasonName]["katz"].values())
        testingDict["closeness"].extend(centralities[seasonName]["closeness"].values())
        testingDict["betweenness"].extend(centralities[seasonName]["betweenness"].values())
      
        # make winners list
        winMarker = []
        currSeasonNum = []
        finalists = seasonFinalists[seasonNum]
        finalistMarker = []
        ages = []
        genders = []
        challengeWinsList = []
        inDegrees = []
        conScores = []
        for contestant in contestants:
            if contestant in challengeWins[seasonNum]:
                challengeWinsList.append(challengeWins[seasonNum][contestant])
            else:
                challengeWinsList.append(0)
            castawaySeasonCont = castawaySeason.loc[castawaySeason["castaway_id"] == contestant]
            castawaySeasonCont 
            age = castawaySeasonCont["age"].item()
            ages.append(age)
            
            currCastaway = castawayDetails.loc[castawayDetails['castaway_id'] == contestant]
            gender = currCastaway["gender"].item()
            if (gender == "Male"):
                genders.append(0)
            else:
                genders.append(1)
            currSeasonNum.append(seasonNum)
            if contestant == winners[seasonNum-1]:
                winMarker.append(1)
            else:
                winMarker.append(0)
            if contestant in finalists:
                finalistMarker.append(1)
            else:
                finalistMarker.append(0)
                '''
            if contestant in candidatesElim[seasonName]:
                episodeElim = candidatesElim[seasonName][contestant]
                elimScore = episodeElim / (lastEpisode+1)
            elif contestant in finalists:
                elimScore = lastEpisode / (lastEpisode+1)
            else:
                elimScore = 5
            winMarker.append(elimScore)'''
            
            contestNum = contestantDict[contestant]
            inDegreeSum = (directedNetworks[seasonName])[:, contestNum].sum()
            inDegrees.append(inDegreeSum)
        for i in range(len(directedNetworks[seasonName])):
            runSum = 0 
            # still need to go through the rows 
            for j in range(len(directedNetworks[seasonName])):
                if i == j:
                    continue 
                
                for k in range(len(directedNetworks[seasonName][0])):
                    runSum += min(directedNetworks[seasonName][i][k], directedNetworks[seasonName][j][k] )
                
            
            conScores.append(runSum)
            
        testingDict["winner"].extend(winMarker)
        testingDict["season"].extend(currSeasonNum)
        testingDict["finalist"].extend(finalistMarker)
        testingDict["age"].extend(ages)
        testingDict["gender"].extend(genders)
        testingDict["challenges_won"].extend(challengeWinsList)
        testingDict["in_degree"].extend(inDegrees)
        testingDict["con_score"].extend(conScores)

# might need to normalize in_degree
maxIndegree = max(fullDict["in_degree"])
fullDict['in_degree'] /= maxIndegree
trainingDict['in_degree'] /= maxIndegree 
testingDict['in_degree'] /= maxIndegree
trainDf = pd.DataFrame(trainingDict)
testDf = pd.DataFrame(testingDict)
fullDf = pd.DataFrame(fullDict)

# can just do the finalists stuff here
finalist_id_list = []
for i in range(1, 44):
    currFinalists = seasonFinalists[i]
    for finalist in currFinalists:
        finalist_id_list.append(finalist)
        
newTrainDf = trainDf.loc[trainDf["contestant"].isin(finalist_id_list)]
#testDf = trainDf.loc[trainDf["contestant"].isin(finalist_id_list)]
#fullDf = trainDf.loc[trainDf["contestant"].isin(finalist_id_list)]

        

# Linear Regression for Winner
model = smf.ols(formula='winner ~ degree + eigenvector + katz + closeness + betweenness', data = trainDf )
res = model.fit()
#print(res.summary())

allFeatures = ["age", "gender", "challenges_won", "degree", "eigenvector", "katz", "closeness", "betweenness", "in_degree", "con_score"]
bestFeatures = ["eigenvector", "closeness", "in_degree", "con_score"]
graphFeatures = ["degree", "eigenvector", "katz", "closeness", "betweenness", "in_degree"]
eigenVector = ["eigenvector"]

x_trainFinalist = trainDf[bestFeatures]
x_trainWinner = newTrainDf[["age", "gender", "degree", "eigenvector", "katz", "closeness", "betweenness"]]
#x_train = trainDf[["degree"]]
x_trainE = trainDf[["eigenvector"]]
y_trainWinner = newTrainDf["winner"]
y_trainWin = trainDf["winner"]
y_trainF = trainDf["finalist"]
LR = LinearRegression()
LR.fit(x_trainFinalist, y_trainWin)

'''
# start to use other features
x_test = testDf[["age", "gender", "degree", "eigenvector", "katz"]]
x_testE = testDf[["eigenvector"]]
y_test = testDf["winner"]
y_testF = testDf["finalist"]
y_pred = LR.predict(x_test)
score = r2_score(y_test, y_pred)
'''
rfRegressorW = RandomForestRegressor(n_estimators=1000, random_state = 0)
rfRegressorF = RandomForestRegressor(n_estimators=1000, random_state = 0)
rfRegressorW.fit(x_trainFinalist, y_trainWin)
rfRegressorF.fit(x_trainFinalist, y_trainF)

xgbCW = XGBRegressor(
    learning_rate = 0.1,
    n_estimators = 1000,
)
xgbCW.fit(x_trainFinalist, y_trainWin)
xgbCF = XGBRegressor(
    learning_rate = 0.1,
    n_estimators = 1000,
)
xgbCF.fit(x_trainFinalist, y_trainF)

# get the finalists for each season, get the centralities and predict
# take the highest probability contestant and see if it equals the actual winner

probabilities = {}
allProbs = {}

# loop through the finalists
verifying = []
verifyingFinalists = []
for season in seasonFinalists:
    #print(season)
    # check the probabilities of all of them
    probabilities[season] = {}
    finalists = seasonFinalists[season]
    #print(finalists)
    
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
    maxProb = -1000
    maxCont = ""
    maxContAll = ""
    # also want to see probability of making it to the finalists
    # get the probabilities of all contestants in a season
    # see if the highest probability contestant made it to the finals
    # another option is training on finalist as the y - ask about this 
    
    checkingFinalists = fullDf.loc[fullDf['season'] == season]
    maxPred = -1000
    for contestant in checkingFinalists['contestant']:
        checkingFinalist = checkingFinalists.loc[checkingFinalists['contestant'] == contestant]
        #currPrediction = res.predict(checkingFinalist[["degree", "eigenvector", "katz", "closeness", "betweenness"]])
        # predicting finalist from the max probability
        
        # checking finalist doesnt have age
        currPrediction = xgbCF.predict(checkingFinalist[bestFeatures])
        
        #currPrediction = rfRegressorF.predict(checkingFinalist[["eigenvector"]])
        if currPrediction > maxPred:
            maxPred = currPrediction
            maxContAll = contestant
    #print(maxContAll)
    if maxContAll in finalists: 
        verifyingFinalists.append(1)
    else:
        verifyingFinalists.append(0)
        
    for finalist in finalists:
        currData = fullDf.loc[fullDf['season'] == season]
        currData = currData.loc[currData['contestant'] == finalist]
        probability = xgbCW.predict(currData[bestFeatures])
        
        # looking for max eigenvector centrality
        #print(centralities[seasonName]["eigenvector"])
        # why only getting up to 12 for this eigenvector one
        seasonName = "season" + str(season)
        #probability = centralities[seasonName]["eigenvector"][contestantDicts[season-1][finalist]]
        
        probabilities[season][finalist] = probability
        if probability > maxProb:
            maxProb = probability
            maxCont = finalist
    if maxCont == winners[season-1]:
        verifying.append(1)
    else:
        verifying.append(0)
        
print("winners predictions") 
print(verifying)
print("finalists predictions")
print(verifyingFinalists)
#print(regressor.feature_importances_)
        
pickle.dump(seasonFinalists, open("finalists.pkl", "wb"))
pickle.dump(contestantDicts, open("contestant_mappings.pkl", "wb"))


# plotting the degree vs elimination episode (actually episode / (total episodes+1)) using matplotlib
# https://matplotlib.org/stable/gallery/shapes_and_collections/scatter.html
# need to turn the contestant into a number so it matches with the degrees output
# should both be a list where the indices match up
degreesAll = []
eigensAll = []
katzAll = []
betweennessAll = []
closenessAll = []
candElimsList = []

for i in range(1,44):
    seasonName = "season" + str(i)
    seasonCandElims = candidatesElim[seasonName]
    degree = centralities[seasonName]["degree"]
    eigen = centralities[seasonName]["eigenvector"]
    katz = centralities[seasonName]["katz"]
    betweenness = centralities[seasonName]["betweenness"]
    closeness = centralities[seasonName]["closeness"]
    degreesAll.extend(degree.values())
    eigensAll.extend(eigen.values())
    katzAll.extend(katz.values())
    betweennessAll.extend(betweenness.values())
    closenessAll.extend(closeness.values())
    
    seasonSummary.loc[seasonSummary["season"]== float(i)]
    winner = seasonWinner["winner"].item()
    
    # collect these throughout the seasons 
    # go through the candidates and get their elimination 
    
    # might just be same order as seasonCandElims
    currNumEps = numEps[i-1]
    
    for cont in contestantDicts[i-1]:
        # this will give the elimEpisode for 
        if cont in seasonCandElims:
            elimEp = seasonCandElims[cont] / (currNumEps+1)
            # divide over total eps + 1
            candElimsList.append(elimEp)
        else:
            if cont == winner:
                candidatesElim[seasonName][cont] = currNumEps 
                elimEp = 1
                candElimsList.append(elimEp)
            else:    
                candidatesElim[seasonName][cont] = currNumEps 
                elimEp = (currNumEps) / (currNumEps+1)
                candElimsList.append(elimEp)
    # need degrees to be a list, also need to factor in finalists

plt.scatter(degreesAll, candElimsList)
plt.savefig('degrees.png')
plt.clf()
plt.scatter(eigensAll, candElimsList)
plt.savefig("eigens.png")
plt.clf()
plt.scatter(katzAll, candElimsList)
plt.savefig("katz.png")
plt.clf()
plt.scatter(betweennessAll, candElimsList)
plt.savefig("betweenness.png")
plt.clf()
plt.scatter(closenessAll, candElimsList)
plt.savefig("closeness.png")
plt.clf()
counter = 0
counterTotal = 0
counterF = 0
counterWTwo = 0
counterFTwo = 0
counterTwoAll = 0
# only looking at test set
for i in range(0, len(verifying), 2):
    counterTotal += 1
    if i in two_finalist_seasons:
        counterTwoAll +=1
        
    if verifying[i] == 1:
        counter += 1
        if i in two_finalist_seasons:
            counterWTwo+=1

    if verifyingFinalists[i] == 1:
        counterF += 1
        if i in two_finalist_seasons:
            counterFTwo+=1
    
    
print(counter)
print(counterF)  
print(counterTotal)
print(counterWTwo)
print(counterFTwo)
print(counterTwoAll)
# store the data in pkl files 
# want both the network matrix and the winners for each season
for i in range(1, 44):
    szn = "season" + str(i)
    pklName = szn + "Merged.pkl"
    winnerPkl = szn + "winner.pkl"
    directedName = szn + "Directed.pkl"
    pickle.dump(aggNet[szn], open(pklName, "wb"))
    pickle.dump(directedNetworks[szn], open(directedName, "wb"))
    # want the number of the winners
    pickle.dump(contestantDicts[i-1][winners[i-1]], open(winnerPkl, "wb"))
    
print(xgbCW.feature_importances_)
print(xgbCF.feature_importances_)


#perm_importance = permutation_importance(rfRegressorW, x_test, y_test)
#print(perm_importance)

A1 = pickle.load(open("season1.pkl", "rb"))
W1 = pickle.load(open("season1winner.pkl", "rb"))
seasonFinalistsNums = {}
for i in range(1,44):
    currFinalistNums = []
    for finalist in seasonFinalists[i]:
        currFinalistNums.append(contestantDicts[i-1][finalist])
    seasonFinalistsNums[i] = currFinalistNums
print(seasonFinalistsNums)
pickle.dump(seasonFinalistsNums, open("finalist_numbers.pkl", "wb"))