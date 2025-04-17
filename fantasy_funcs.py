################
def newcalcrate(vec1, vec2):
## list, list -> int
## list of 0/1 from one shared words from each title -> score
    import numpy as np

    
    bindexes = [x for x in range(len(vec1)) if vec1[x] != 0 or vec2[x] != 0]#list of indexes that are included in either title
    lindexes = [x for x in range(len(vec1)) if vec1[x] != 0 and vec2[x] != 0]#list of indexes that are conjunctions of both lists
    if len(lindexes) == 0: # if there is no overlapping indexes the score is 0
        score = 0
    else: 
        numincld = len(lindexes)/len(bindexes) # ratio of words that are shared/ possible words that could be shared
        # find where each index is for each vector
        inda = vec1[lindexes]
        indb = vec2[lindexes]
        
        # calculate how far apart are shared words in each title. Lower is letter
        numpotential = 1/np.mean(np.abs(inda-indb)+1) #average differences between word places in each titles + 1 so you overlapping words become 1/1 not 1/0
    
        checkset = set(np.abs(inda-indb))#if they have two words that are in the same place for BOTH titles, then give them a bonus for extra shared words
        if len(inda) == 1: # if its just one word thats in both titles, they literally have the same name e.g. Pikmin and Pikmin 2 (the 2 will have been removed and it becomes just Pikmin)
            bonus = 1
        elif len(checkset) < len(inda): # the checkset is less than the length of inda meaning two words are shared in the same place in two titles
            bonus = 1
        else:
            bonus = 0
        score = (numincld + numpotential + bonus)/3 # find the average of each metric and return
    
    return(score)


def generate_clusters(names, platforms = None, drawhuh = False, drawwindow = (0,0)):
    # list, bool, tuple -> list, image
    # take a list of title names and returns list of their indexes from the initial list clustered
    
    import numpy as np
    from collections import Counter
    import random
    
    
    
    
    ##CLEAN THE TITLES
    
    # dead words we are excluding
    dead_words = ['The','of','the', 'in', 'In','and','World','of', 'for', 'Edition',"Collector's", '&', '2','3','4','5','6','7','8','9','10', 'a','A', 'On','on', 'dlc', 'u', '', ' ', '+', '-','&', 'I','II','III','IV','V','VI','VII','VIII','IX','X']

    if platforms:
        dead_words += platforms
    
    # lets split the names up into there components -> Mario Kart : ['Mario','Kart']
    splitnames =[x.split() for x in names]
    
    # next lets remove any words that are meaningless and might inflate similarity.
    lilnames = []# should be a list of list of words
    for i in range(len(splitnames)):
        lilnames.insert(i,[ y.translate({ord(k): None for k in ":$@&/!.+-$#~'"}).lower() for y in splitnames[i] if y not in dead_words])
    
    # also lets just get a list of all the words that are used for all titles. The goal is to count how many times different words are used so we want them all in one big soup
    all_names = [z  for y in lilnames for z in y if z not in dead_words]
    
    counts = Counter(all_names) #generate the frequency counts
    
    # let's get rid of all the ones that only have 1 in them
    set_names = list(set(all_names)) # get all the uniques ones which are also the keys for the counts
    pot_titles = [nm for nm in set_names if counts[nm] > 1 and nm != ''] # include only the words that appear in at least two different titles and are not ''
    
    
    
    
    ## FORMAT THE MATRICES
    
    ## now we are going to create a vector of 1/0 based on if the words in the list are in the pot_titles. The matrix (name_count) is one row for each title name and one column for each word in the pot_titles
    name_count = np.zeros((len(lilnames),len(pot_titles))) # starting with a 0 matrix
    
    ## assign the values for the matrix
    for iname in range(len(lilnames)):# for each name
        for iword in range(len(lilnames[iname])): # for each word
            if lilnames[iname][iword] in pot_titles: # if its in the list of potential matching words
                whereloc = pot_titles.index(lilnames[iname][iword]) #find where the match is 
                name_count[iname, whereloc] = iword + 1 #assign it to its location plus 1!
    
    
    
    
    ## GENERATE METRIC MATRICES
    
    ## next we are going to calculate an index of how similar the vectors for each pair of titles are to each other. This will call the newcalcrate function also in this file. The final shape is a 
    #len(lilnames) x len(lilnames)
    
    indexscores = np.zeros((name_count.shape[0], name_count.shape[0]))
    for irow in range(name_count.shape[0]):
        for icol in range(name_count.shape[0]):
            indexscores[icol, irow] = newcalcrate(name_count[irow], name_count[icol])
    
    ## we are now going to use a cutoff score to make later correlations clearer. The cutoff is going to dichotomize the scores
    cutoff = .75
    splitdiff = indexscores
    splitdiff[splitdiff >=cutoff] = 1
    splitdiff[splitdiff < cutoff] = 0
    
    # now lets take the correlation of the similarity patterns
    corrmat = np.corrcoef(splitdiff)
    corrmat[corrmat > .75] = 1
    
    if drawhuh: # if you want to draw the figures draw them now
        import matplotlib.pyplot as plt
        import seaborn as sns

        if drawwindow == (0,0):
            drawstart = 0
            drawstop = len(lilnames)
        else:
            drawstart = drawwindow[0]
            drawstop = drawwindow[1]
    
        sns.heatmap(splitdiff[drawstart:drawstop], linewidth=0.5,xticklabels = lilnames[drawstart:drawstop], yticklabels = lilnames[drawstart:drawstop]).set_title("splitdiff")
        plt.show()
    
        sns.heatmap(indexscores[drawstart:drawstop], linewidth=0.5,xticklabels = lilnames[drawstart:drawstop], yticklabels = lilnames[drawstart:drawstop]).set_title("indexscores")
        plt.show()
    
        sns.heatmap(corrmat[drawstart:drawstop], linewidth=0.5,xticklabels = lilnames[drawstart:drawstop], yticklabels = lilnames[drawstart:drawstop]).set_title("corrmat")
        plt.show()
        
    
    
    ## GROUPING INTO CLUSTERS
    unclustered = [] # eventual home of games that don't have partners
    clusters = [] # eventual home of games in clusters
    perfectIt = np.array([x for x in range(corrmat.shape[0]) if sum(corrmat[x,:] == 1)>1]) # find all the ones that have 'perfect correlations' with at least one other game
    setAll = set(list(range(splitdiff.shape[0]))) # make the list of all indexes a set so we can find whats not there
    notperfect = list(setAll.difference(perfectIt)) # all the ones not in the perfectCoor
    extraAdd = [] # sometimes groups will be randomly added so lets make sure that happens at the end so it doesn't influence other ones
    
    used = np.empty(0) # indexes that have already been added to a cluster 
    
    # THE PERFECT CORRELATION LIST
    # start looping through all the indexes in the perfectIt. They should slowly be getting removed
    while len(perfectIt)> 0:
        i = perfectIt[0] # current index
        matches = [x for x in range(len(corrmat[i,:])) if corrmat[i,x] == 1] #find all the perfect matches 
        overlap = [x for x in matches if x not in used] # make sure they aren't in another cluster already
        if len(overlap) == len(matches): # if all of the matches haven't been used before...
            clusters.append(matches) # add a new cluster
            perfectIt = [x for x in perfectIt if x not in matches] # remove them from the perfect It
            used = np.concatenate((used, np.array(matches))) # list them as used 
        else: # then its perfect matches are already in another cluster!
            # count the number of matches in the already existing clusters if they are greater than 1
            posclust = [[clust, len(list(set(clusters[clust]).intersection(set(matches))))] for clust in range(len(clusters)) if len(list(set(clusters[clust]).intersection(set(matches))))>0]
            counts = np.array(posclust) # turn posclust into np.array for use of np.where
            most = np.where(counts[:,1] == max(counts[:,1])) #where is the maximum of identical correlations. There should only ever be one I think but its a double check
            if len(most[0]) == 1:
                clusters[most[0][0]].append(i) # add to its cluster
            else:
                #randomly choose 
                clusters[random.choice(most[0])].append(i) # if there are multiple groups with the same number of matches, randomly assign 
            perfectIt = [x for x in perfectIt if x not in [i]] # remove them from the perfect It
            used = np.append(used, i)#((used, np.array(matches))) # list them as used 
    
    
    
    # FOR ONES THAT DO NOT HAVE PERFECT MATCHES
    # let's see we can get ones with a 'good fit'
    for k in range(len(notperfect)):
        holder = corrmat[notperfect[k],:] # find all their correlations
        holder[notperfect[k]] = 0 # get rid of its correlation to itself which is either 1 or nan
        maxval= max(holder) # where is the highest value that is not to itself
        if np.isnan(maxval) or maxval <= .1: # if its a nan value (it has no matches) or below .1 lets exclude it from matching with something
            unclustered.append(notperfect[k])
        else: #okay so if there is a max value
            theloc = np.where(holder == maxval) #find if it already exists in a cluster
            posclust = [[clust, len(list(set(clusters[clust]).intersection(set(theloc[0]))))] for clust in range(len(clusters)) if len(list(set(clusters[clust]).intersection(set(theloc[0]))))>0]
            if len(posclust) == 0: # if no cluster exists with it but another one somewhere will have a pairing with it, lets make a new cluster
                clusters.append([notperfect[k]])
            elif len(posclust) == 1: # if one cluster matches, add it
                #print(names[notperfect[k]], 'matches with ', posclust)
                clusters[posclust[0][0]].append(notperfect[k])
            else:#since there is more than one, let's put it in the cluster with the most matching
                #print(names[notperfect[k]], 'matches with ', posclust)
                counts = np.array(posclust)
                most = np.where(counts[:,1] == max(counts[:,1]))
                if len(most[0]) == 1:
                    clusters[most[0][0]].append(notperfect[k])
                else:
                    extraAdd.append(notperfect[k])
                    #randomly choose

    for l in range(len(extraAdd)):
        holder = corrmat[extraAdd[l],:] # find all their correlations
        holder[extraAdd[l]] = 0 # get rid of its correlation to itself which is either 1 or nan
        maxval= max(holder) # where is the highest value that is not to itself
        theloc = np.where(holder == maxval) #find if it already exists in a cluster
        posclust = [[clust, len(list(set(clusters[clust]).intersection(set(theloc[0]))))] for clust in range(len(clusters)) if len(list(set(clusters[clust]).intersection(set(theloc[0]))))>0]
        if len(posclust) == 0: # if no cluster exists with it but another one somewhere will have a pairing with it, lets make a new cluster
            clusters.append([extraAdd[l]])
        elif len(posclust) == 1: # if one cluster matches, add it
            clusters[posclust[0][0]].append(extraAdd[l])
        else:#since there is more than one, let's put it in the cluster with the most matching
            counts = np.array(posclust)
            most = np.where(counts[:,1] == max(counts[:,1]))
            if len(most[0]) == 1:
                clusters[most[0][0]].append(extraAdd[l])
            else:
                #randomly choose
                clusters[random.choice(most[0])].append(extraAdd[l])
        
    

    return(clusters, unclustered)
