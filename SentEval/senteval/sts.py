# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
STS-{2012,2013,2014,2015,2016} (unsupervised) and
STS-benchmark (supervised) tasks
'''

from __future__ import absolute_import, division, unicode_literals

import os
import io
import numpy as np
import logging
import csv
import json
import pandas as pd
from tqdm import tqdm

from scipy.stats import spearmanr, pearsonr

from senteval.utils import cosine
from senteval.sick import SICKEval


class STSEval(object):
    def loadFile(self, fpath):
        self.data = {}
        self.samples = []

        for dataset in self.datasets:
            sent1, sent2 = zip(*[l.split("\t") for l in
                               io.open(fpath + '/STS.input.%s.txt' % dataset,
                                       encoding='utf8').read().splitlines()])
            raw_scores = np.array([x for x in
                                   io.open(fpath + '/STS.gs.%s.txt' % dataset,
                                           encoding='utf8')
                                   .read().splitlines()])
            not_empty_idx = raw_scores != ''

            gs_scores = [float(x) for x in raw_scores[not_empty_idx]]
            sent1 = np.array([s.split() for s in sent1])[not_empty_idx]
            sent2 = np.array([s.split() for s in sent2])[not_empty_idx]
            # sort data by length to minimize padding in batcher
            sorted_data = sorted(zip(sent1, sent2, gs_scores),
                                 key=lambda z: (len(z[0]), len(z[1]), z[2]))
            sent1, sent2, gs_scores = map(list, zip(*sorted_data))

            self.data[dataset] = (sent1, sent2, gs_scores)
            self.samples += sent1 + sent2

    def do_prepare(self, params, prepare):
        if 'similarity' in params:
            self.similarity = params.similarity
        else:  # Default similarity is cosine
            self.similarity = lambda s1, s2: np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))
        return prepare(params, self.samples)

    def run(self, params, batcher):
        results = {}
        all_sys_scores = []
        all_gs_scores = []
        for dataset in self.datasets:
            sys_scores = []
            input1, input2, gs_scores = self.data[dataset]
            for ii in range(0, len(gs_scores), params.batch_size):
                batch1 = input1[ii:ii + params.batch_size]
                batch2 = input2[ii:ii + params.batch_size]

                # we assume get_batch already throws out the faulty ones
                if len(batch1) == len(batch2) and len(batch1) > 0:
                    enc1 = batcher(params, batch1)
                    enc2 = batcher(params, batch2)

                    for kk in range(enc2.shape[0]):
                        sys_score = self.similarity(enc1[kk], enc2[kk])
                        sys_scores.append(sys_score)
                        
            ###################################################################
            # do bootstrap for cococxc dataset.
            if self.datasets == ['test']:
                model_prediction = pd.DataFrame({'gs_scores': gs_scores, 'sys_scores': sys_scores})
                cxc_dcpcse = pd.concat([self.cxc_data, model_prediction[['sys_scores']]], axis=1)
                
                np.random.seed(123)
                spearmanr_all = []
                logging.debug('Start bootstrap\n')
                for _ in tqdm(range(1000)):  #so B=1000
                    pop = range(len(self.coco_test))
                    sample = np.random.choice(pop, size=len(self.coco_test)//2) #so n=2500
                    
                    caption1 = []
                    caption2 = []
                    for i in sample:
                        n_sent_pair = len(self.coco_test[i]['cxc_scores'])
                        idx = np.random.choice(n_sent_pair, size=1)[0]
                        caption1.append(self.coco_test[i]['cxc_scores'][idx][0])
                        caption2.append(self.coco_test[i]['cxc_scores'][idx][1])
                    idx_df = pd.DataFrame({'caption1':caption1, 'caption2': caption2})
                    
                    evaluation = pd.merge(idx_df, cxc_dcpcse, on=['caption1', 'caption2'], how='left')
                    spearmanr_all.append(spearmanr(evaluation.gs_scores, evaluation.sys_scores)[0])
                    
                logging.debug('%s : bootstrap spearman = %.3f, std = %.3f' %
                              (dataset, np.mean(spearmanr_all),
                           np.std(spearmanr_all)))
            ###################################################################
            
            all_sys_scores.extend(sys_scores)
            all_gs_scores.extend(gs_scores)
            results[dataset] = {'pearson': pearsonr(sys_scores, gs_scores),
                                'spearman': spearmanr(sys_scores, gs_scores),
                                'nsamples': len(sys_scores)}
            logging.debug('%s : pearson = %.4f, spearman = %.4f' %
                          (dataset, results[dataset]['pearson'][0],
                           results[dataset]['spearman'][0]))

        weights = [results[dset]['nsamples'] for dset in results.keys()]
        list_prs = np.array([results[dset]['pearson'][0] for
                            dset in results.keys()])
        list_spr = np.array([results[dset]['spearman'][0] for
                            dset in results.keys()])

        avg_pearson = np.average(list_prs)
        avg_spearman = np.average(list_spr)
        wavg_pearson = np.average(list_prs, weights=weights)
        wavg_spearman = np.average(list_spr, weights=weights)
        all_pearson = pearsonr(all_sys_scores, all_gs_scores)
        all_spearman = spearmanr(all_sys_scores, all_gs_scores)
        results['all'] = {'pearson': {'all': all_pearson[0],
                                      'mean': avg_pearson,
                                      'wmean': wavg_pearson},
                          'spearman': {'all': all_spearman[0],
                                       'mean': avg_spearman,
                                       'wmean': wavg_spearman}}
        logging.debug('ALL : Pearson = %.4f, \
            Spearman = %.4f' % (all_pearson[0], all_spearman[0]))
        logging.debug('ALL (weighted average) : Pearson = %.4f, \
            Spearman = %.4f' % (wavg_pearson, wavg_spearman))
        logging.debug('ALL (average) : Pearson = %.4f, \
            Spearman = %.4f\n' % (avg_pearson, avg_spearman))

        return results


class STS12Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS12 *****\n\n')
        self.seed = seed
        self.datasets = ['MSRpar', 'MSRvid', 'SMTeuroparl',
                         'surprise.OnWN', 'surprise.SMTnews']
        self.loadFile(taskpath)


class STS13Eval(STSEval):
    # STS13 here does not contain the "SMT" subtask due to LICENSE issue
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS13 (-SMT) *****\n\n')
        self.seed = seed
        self.datasets = ['FNWN', 'headlines', 'OnWN']
        self.loadFile(taskpath)


class STS14Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS14 *****\n\n')
        self.seed = seed
        self.datasets = ['deft-forum', 'deft-news', 'headlines',
                         'images', 'OnWN', 'tweet-news']
        self.loadFile(taskpath)


class STS15Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS15 *****\n\n')
        self.seed = seed
        self.datasets = ['answers-forums', 'answers-students',
                         'belief', 'headlines', 'images']
        self.loadFile(taskpath)


class STS16Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS16 *****\n\n')
        self.seed = seed
        self.datasets = ['answer-answer', 'headlines', 'plagiarism',
                         'postediting', 'question-question']
        self.loadFile(taskpath)


class STSBenchmarkEval(STSEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : STSBenchmark *****\n\n')
        self.seed = seed
        self.samples = []
        train = self.loadFile(os.path.join(task_path, 'sts-train.csv'))
        dev = self.loadFile(os.path.join(task_path, 'sts-dev.csv'))
        test = self.loadFile(os.path.join(task_path, 'sts-test.csv'))
        self.datasets = ['train', 'dev', 'test']
        self.data = {'train': train, 'dev': dev, 'test': test}

    def loadFile(self, fpath):
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('\t')
                sick_data['X_A'].append(text[5].split())
                sick_data['X_B'].append(text[6].split())
                sick_data['y'].append(text[4])

        sick_data['y'] = [float(s) for s in sick_data['y']]
        self.samples += sick_data['X_A'] + sick_data["X_B"]
        return (sick_data['X_A'], sick_data["X_B"], sick_data['y'])

class STSBenchmarkFinetune(SICKEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : STSBenchmark *****\n\n')
        self.seed = seed
        train = self.loadFile(os.path.join(task_path, 'sts-train.csv'))
        dev = self.loadFile(os.path.join(task_path, 'sts-dev.csv'))
        test = self.loadFile(os.path.join(task_path, 'sts-test.csv'))
        self.sick_data = {'train': train, 'dev': dev, 'test': test}

    def loadFile(self, fpath):
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('\t')
                sick_data['X_A'].append(text[5].split())
                sick_data['X_B'].append(text[6].split())
                sick_data['y'].append(text[4])

        sick_data['y'] = [float(s) for s in sick_data['y']]
        return sick_data
        
class SICKRelatednessEval(STSEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : SICKRelatedness *****\n\n')
        self.seed = seed
        self.samples = []
        train = self.loadFile(os.path.join(task_path, 'SICK_train.txt'))
        dev = self.loadFile(os.path.join(task_path, 'SICK_trial.txt'))
        test = self.loadFile(os.path.join(task_path, 'SICK_test_annotated.txt'))
        self.datasets = ['train', 'dev', 'test']
        self.data = {'train': train, 'dev': dev, 'test': test}
    
    def loadFile(self, fpath):
        skipFirstLine = True
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if skipFirstLine:
                    skipFirstLine = False
                else:
                    text = line.strip().split('\t')
                    sick_data['X_A'].append(text[1].split())
                    sick_data['X_B'].append(text[2].split())
                    sick_data['y'].append(text[3])

        sick_data['y'] = [float(s) for s in sick_data['y']]
        self.samples += sick_data['X_A'] + sick_data["X_B"]
        return (sick_data['X_A'], sick_data["X_B"], sick_data['y'])

###############################################################################
class CocoCXCEval(STSEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : CocoCXC *****\n\n')
        self.seed = seed
        self.samples = []
        test = self.loadFile(os.path.join(task_path, 'cxc-sts-test.csv'))
        self.datasets = ['test']
        self.data = {'test':test}
        self.coco_test = self.load_coco_test(os.path.join(task_path, 'dataset_coco_with_cxc.json'))
        self.cxc_data = pd.read_csv(os.path.join(task_path, 'cxc-sts-test.csv')) \
            [['caption1', 'caption2', 'agg_score']].rename(columns={'agg_score':'gs_scores'})
        
    def loadFile(self, fpath):
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        
        reader = csv.reader(open(fpath, "r", encoding='utf-8'), delimiter=",")
        next(reader)  # Skip header.
        for row in reader:
            caption1, caption2, score, rating_type, sent1, sent2 = row
            sick_data['X_A'].append(sent1.split())
            sick_data['X_B'].append(sent2.split())
            sick_data['y'].append(score)

        sick_data['y'] = [float(s) for s in sick_data['y']]
        self.samples += sick_data['X_A'] + sick_data["X_B"]
        return (sick_data['X_A'], sick_data["X_B"], sick_data['y'])
    
    def load_coco_test(self, path):
        data = json.load(open(path, "r"))
        test_data = []
        for i in data['images']:
            if i['split'] == 'test':
                test_data.append(i)
        return test_data
###############################################################################
