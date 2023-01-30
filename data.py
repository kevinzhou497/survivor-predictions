import pyreadr
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# retrieving vote history as a panda dataframe
vote_history = pyreadr.read_r('vote_history.rda')
voteHistory = vote_history["vote_history"]

# only keep the US seasons
voteHistory = voteHistory.loc[voteHistory['version'] == 'US']
# voteHistory.to_csv("vote_history.csv")
lastSeason = int(voteHistory["season"].max())

# have a list of dataframes
seasons = []
# have a loop that goes through seasons and appends to the list of dataframes
for i in range(1, 44):
    seasonNum = float(i)
    seasonVotes = voteHistory.loc[voteHistory["season"] == seasonNum]
    seasons.append(seasonVotes)
# make a matrix for each contestant for each episode of each season
networks = {}
aggNet = {}

seasonNum = 0
contestantDicts = []
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
    
    # go through each episode
    
    seasonName = "season" + str(seasonNum)
    networks[seasonName] = {}
    aggNet[seasonName] = {}
    aggSeason = []
    for i in range(numContestants):
                aggSeason.append([0 for i in range(numContestants)])
    aggSeason = np.asarray(aggSeason)

    for episode in range(lastEpisode+1):
        episodeVotes = season.loc[season["episode"] == float(episode)]
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
                
            # if i and j both voted for this contestant, A_{ij} = 1
            
            contestantNet = np.asarray(contestantNet)
            
        
            # everyone in this list should be marked 1 with each other in the matrix
            votedFor = episodeVotes.loc[episodeVotes["vote"] == contestant]["castaway"].unique()
            
            # can make this more efficient
            for person in votedFor:
                for person2 in votedFor:
                    if person == person2:
                        continue
                    contestantNet[contestantDict[person]][contestantDict[person2]] = 1
                    # maybe doing this too much? need to have it for each person, aggregated wouldnt have it for each person
                    aggEpisode += contestantNet
      
            G = nx.from_numpy_array(contestantNet)
            
            # tuple with adjacency matrix and the networkx graph
            networks[seasonName][episode][contestant] = (contestantNet, G)
        
        # aggregate for each episode
        # print(aggEpisode)
        aggNet[seasonName][episode] = (aggEpisode, nx.from_numpy_array(aggEpisode))
        aggSeason += aggEpisode
    
    aggNet[seasonName] = (aggSeason, nx.from_numpy_array(aggSeason))

            
# print(networks)
# print(aggNet["season43"])
nx.draw(aggNet["season43"][1])
plt.show()