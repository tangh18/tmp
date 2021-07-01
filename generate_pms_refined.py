# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sample Generate GPT2"""
open_old_pronounce = 1
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import time
from datetime import datetime
from arguments import get_args
from utils import Timers
from pretrain_gpt2 import initialize_distributed
from pretrain_gpt2 import set_random_seed
from pretrain_gpt2 import get_train_val_test_data
from pretrain_gpt2 import get_masks_and_position_ids
from utils import load_checkpoint, get_checkpoint_iteration
from data_utils import make_tokenizer
from configure_data import configure_data
import mpu
import deepspeed
import copy
from fp16 import FP16_Module
from model import GPT2Model
from model import DistributedDataParallel as DDP
from utils import print_rank_0
from pretrain_gpt2 import get_model
from pypinyin import pinyin, FINALS, FINALS_TONE, TONE3
import jsonlines
import torch.multiprocessing as mp
import threading


def generate():
    fi = []
    print("generate begin")
    title_list = [["咏特朗普", "咏美国前总统，民粹主义政客唐纳德 特朗普"]]

    author_list = ["李白", "杜甫"]

    for j in author_list:
        for i in title_list:
            fi.append([i[0], j, i[1]])
            # fi.append([i[0],j,i[1]])

    # with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=True, profile_memory=False) as prof:
    #import time
    #prof.export_chrome_trace('../repo2/' + str(time.time())+'zhiyuan_profile.json')
    print("begin generate_strs")
    time1 = time.time()
    output = generate_strs(fi)
    time2 = time.time()
    print("generate_strs time:", time2 - time1)
    return 0


def generate_strs(tups):
    model, tokenizer, args = prepare_model()
    print("prepare_model finished")
    print("tups {}".format(tups))
    output = []
    print("generate_strs tups num:", len(tups))
    for tup in tups:
        #str = generate_token_tensor(str,tokenizer)
        print("generate_strs tup:", tup)

        time1 = time.time()
        output_string, output_scores = generate_string(
            model, tokenizer, args, torch.cuda.current_device(), tup[0], tup[1], tup[2])
        time2 = time.time()
        print("generate_string time:", time2 - time1)

        ranklist = np.argsort(output_scores)
        best_score = output_scores[ranklist[0]]
        text_dir = "poems_save/"
        already = []
        '''
        with jsonlines.open(text_dir+tup[0]+tup[1]+'.jsonl', mode='w') as writer:
            for i in range(len(ranklist)):
                j=ranklist[i]
                if output_scores[j]<best_score+2:
                    if not(output_string[j][0:15] in already):
                        otc={}
                        otc['author']=tup[1]
                        otc['title']=tup[0]
                        otc['context']=output_string[j]
                        #print(otc)
                        writer.write(otc)
                        already.append(output_string[j][0:15])
        '''
    return 0


def generate_string(model, tokenizer, args, device, title, author, desc=None, length=None):
    input_str = title + " 作者:" + author + " 体裁:诗歌 题名:" + title + " 正文: "
    if desc is not None:
        input_str = title + " 作者:" + author + " 体裁:诗歌 描述:" + desc + " 题名:" + title + " 正文: "
    #aus = author.split(' ')[1]
    input_len = len(input_str)
    context_count = 0
    print("model eval")
    print("model time eval end")
    with torch.no_grad():
        context_tokens = tokenizer.EncodeAsIds(input_str).tokenization
        eo_tokens = tokenizer.EncodeAsIds('<|endoftext|>').tokenization
        context_length = len(context_tokens)
        if context_length >= args.seq_length:
            return 0, "输入过长。"

        context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
        eo_token_tensor = torch.cuda.LongTensor(eo_tokens)
        context_length_tensor = torch.cuda.LongTensor([context_length])
        context_length = context_length_tensor[0].item()

        counter, mems = 0, []
        org_context_length = context_length
        beam_size = 15
        beam_candidate = 7
        beam_max = 2
        max_headings = 6
        final_storage = []
        final_storage_score = []
        step = 9

        time_sentence_start = time.time()
        if length is None:
            beam_sentences = generate_sentence(
                model, tokenizer, args, device, context_tokens_tensor, [], num_candidates=beam_size * 5)
        if length == 5:
            beam_sentences = generate_sentence(
                model, tokenizer, args, device, context_tokens_tensor, [], num_candidates=beam_size * 5, max_length=6)
        if length == 7:
            beam_sentences = generate_sentence(
                model, tokenizer, args, device, context_tokens_tensor, [], num_candidates=beam_size * 5, min_length=6)
        time_sentence_end = time.time()
        print("time sentence {}".format(time_sentence_end - time_sentence_start))

        ## batch process begin
        time1 = time.time()
        overall_score = []
        past_beam_id = []
        input_str_batch = []
        output_str_batch = []
        for w in range(len(beam_sentences)):
            if '<|end' in beam_sentences[w][0]:
                continue
            input_str = '”' + beam_sentences[w][0] + '”此句出自'
            output_str = '古诗《' + title + '》'
            input_str_batch.append(input_str)
            output_str_batch.append(output_str)

        # print("generate_string generate_score batch inputs:", input_str_batch, output_str_batch)
        scores = generate_score_batch(model, tokenizer, args, device, input_str_batch, output_str_batch)
        pos = 0
        for w in range(len(beam_sentences)):
            if '<|end' in beam_sentences[w][0]:
                continue
            ss = -beam_sentences[w][1] / len(beam_sentences[w][0]) - 8
            # print("ss:", type(ss), ss)
            # ss = ss.cpu().numpy()
            iscore = scores[pos] - 0.45 * (np.abs(ss) + ss)
            pos += 1
            beam_sentences[w][1] = iscore
            overall_score.append(iscore)
            past_beam_id.append(w)

        time2 = time.time()
        print("generate_string generate_score batch time:", time2 - time1)
        # print("generate_string generate_score overall_score:", overall_score)

        gy = np.argsort(overall_score)
        k = 0
        sumbeam = np.zeros(100)
        time_beam_for_end = time.time()
        print("beam time {}".format(time_beam_for_end - time_sentence_end))
        gym = []
        num = 0
        while (num < beam_size) and (k <= len(gy)):
            k += 1
            if sumbeam[past_beam_id[gy[-k]]] < beam_max:
                sumbeam[past_beam_id[gy[-k]]] += 1
                gym.append(gy[-k])
                num += 1
        best_score = -1000
        best_pos = 0

        print("generate_string step:", step)
        for i in range(step):
            time_modes = 0
            if (best_score > -1000) and (i > 8):
                del beam_sentences
                del beam_new_sentences
                torch.cuda.empty_cache()
                return final_storage, final_storage_score
            beam_new_sentences = []

            endnote = [',', '，', '?', '？']
            if i % 2 == 0:
                endnote = ['。', '?', '？', '！', '!']
            size = beam_size
            if len(gym) < size:
                size = len(gym)
            if size == 0:
                del beam_sentences
                del beam_new_sentences
                torch.cuda.empty_cache()
                return final_storage, final_storage_score
            ini_score = beam_sentences[gym[0]][1] / (i + 1)
            # early stopping
            if i > 7:
                ini_score -= 0.2
            if i > 11:
                ini_score -= 0.4

            if ini_score < best_score - 2:
                del beam_sentences
                del beam_new_sentences
                torch.cuda.empty_cache()
                return final_storage, final_storage_score

            overall_score = []
            past_beam_id = []

            id_batch = []
            gen_batch = []
            current_sentence_batch = []
            ini_score_batch = []

            token_tensor_batch = []
            mems_batch = []
            len_sentence_batch = []

            print("generate_string loop size:", size)
            # batch process data of all loops
            step_time1 = time.time()
            for w in range(size):
                id = gym[w]
                current_sentence = input_str + beam_sentences[id][0]
                ini_score = beam_sentences[id][1]
                token_tensor = beam_sentences[id][2]
                mems = beam_sentences[id][3]
                len_sentence = getlength(beam_sentences[id][0])

                id_batch.append(id)
                current_sentence_batch.append(current_sentence)
                ini_score_batch.append(ini_score)

                token_tensor_batch.append(token_tensor)
                mems_batch.append(mems)
                len_sentence_batch.append(len_sentence)

            time1 = time.time()
            group_num = 8
            token_tensor_batch0 = token_tensor_batch[:group_num]
            mems_batch0 = mems_batch[:group_num]
            len_sentence_batch0 = len_sentence_batch[:group_num]

            token_tensor_batch1 = token_tensor_batch[group_num:]
            mems_batch1 = mems_batch[group_num:]
            len_sentence_batch1 = len_sentence_batch[group_num:]

            # infer twice to reduce memory usage
            gen_batch = generate_sentence_batch(model, tokenizer, args, device, token_tensor_batch0, mems_batch0, len_sentence_batch0,
                                                num_candidates=beam_candidate, endnote=endnote)
            time2 = time.time()
            print("generate_sentence_batch time 0:", time2 - time1)

            gen_batch1 = generate_sentence_batch(model, tokenizer, args, device, token_tensor_batch1, mems_batch1, len_sentence_batch1,
                                                num_candidates=beam_candidate, endnote=endnote)
            gen_batch.extend(gen_batch1)
            time3 = time.time()
            print("generate_sentence_batch time 1:", time3 - time1, time2 - time1)

            # get score by batch
            for w in range(size):#改成多GPU并行
                id = id_batch[w]
                gen = gen_batch[w]
                current_sentence = current_sentence_batch[w]
                ini_score = ini_score_batch[w]

                score_time1 = time.time()
                best_score, best_pos = get_score_after_sentence_batch(model, tokenizer, args, device, i, id, w, title, 
                                                                      current_sentence, input_len, ini_score, best_score, best_pos, 
                                                                      gen, final_storage, beam_sentences, final_storage_score, past_beam_id, beam_new_sentences, overall_score)
                score_time2 = time.time()
                # print("get_score_after_sentence batch time:", score_time2 - score_time1)

            step_time2 = time.time()
            print("step time:", step, step_time2-step_time1)
            del beam_sentences
            torch.cuda.empty_cache()
            beam_sentences = beam_new_sentences
            gy = np.argsort(overall_score)
            sumbeam = np.zeros(100)
            sumheading = {}
            k = 0
            gym = []
            num = 0
            while (num < beam_size) and (k + 1 < len(past_beam_id)):
                k += 1
                if sumbeam[past_beam_id[gy[-k]]] < beam_max:
                    wd = beam_sentences[gy[-k]][0][:5]

                    if (not(wd in sumheading)) or (sumheading[wd] < max_headings):
                        if not(wd in sumheading):
                            sumheading[wd] = 1
                        else:
                            sumheading[wd] += 1
                        sumbeam[past_beam_id[gy[-k]]] += 1
                        gym.append(gy[-k])
                        num += 1

        del beam_sentences
        del beam_new_sentences
        torch.cuda.empty_cache()

        return final_storage, final_storage_score


def cat_tensors(tensors):
    if len(tensors)==1:
        return tensors[0]
    return torch.cat(tensors, 0)


def get_mems_batch(mems_batch):
    batch_num = len(mems_batch)
    if batch_num == 1:
        return mems_batch[0]
    mems_new = []
    mems_len = len(mems_batch[0])
    for i in range(mems_len):
        elem_tensors = [mems_batch[j][i] for j in range(batch_num)]
        cat_tensor = torch.cat(elem_tensors, 0)
        mems_new.append(cat_tensor)
    return mems_new


def split_mems_batch(mems_all):
    mems_len = len(mems_all)
    batch_num = mems_all[0].size(0)
    mems_batch = [[] for i in range(batch_num)]
    for i in range(mems_len):
        elem = mems_all[i]
        elems = torch.split(elem, 1)
        for bcnt in range(len(elems)):
            mems_batch[bcnt].append(elems[bcnt])
    return mems_batch


def get_curvote(endnote):
    curvote = 1
    if ',' in endnote:
        curvote = 0
    return curvote


def generate_sentence_batch(model, tokenizer, args, device, tokens_batch, mems_batch, length_batch, endnote=[",", "，", "?", "？"], num_candidates=10):
    '''
    called by generate_string
    only process the condition that min_length == max_length
    '''
    batch_num = len(tokens_batch)
    print("generate_sentence_batch num:", batch_num)

    logits_batch = []
    rts_batch = []

    time1 = time.time()
    for bcnt in range(batch_num):
        tokens = tokens_batch[bcnt]
        index = len(tokens[0])
        in_tokens = tokens[:, index - 1: index]
        in_position_ids = tokens.new_ones((1, 1)) * (index - 1)
        in_attention_mask = tokens.new_ones(1, 1, 1, args.mem_length + 1, device=device, dtype=torch.float)
        mems = mems_batch[bcnt]

        # print("generate_sentence_batch model input:", bcnt, in_tokens.shape, mems[0].shape)
        logits_ret, *rts_ret = model(in_tokens, in_position_ids, in_attention_mask, *mems)
        
        logits_batch.append(logits_ret)
        rts_batch.append(rts_ret)

    time2 = time.time()
    print("generate_sentence_batch model time:", time2 - time1)

    time1 = time.time()
    final_result = generate_sentence_postprocess(model, tokenizer, args, device, endnote, tokens_batch, logits_batch, rts_batch, length_batch, num_candidates)
    time2 = time.time()
    print("generate_sentence_postprocess time:", time2 - time1)

    logits_batch.clear()
    rts_batch.clear()
    torch.cuda.empty_cache()

    return final_result


def generate_sentence_postprocess(model, tokenizer, args, device, endnote, tokens_batch, logits_batch, rts_batch, length_batch, num_candidates):
    batch_num = len(tokens_batch)
    max_tries = num_candidates * 30
    # print("max_tries:", max_tries)

    original_context_batch = []
    context_length_batch = []
    mct_tree_batch = {}

    for bcnt in range(batch_num):
        tokens = tokens_batch[bcnt]
        logits = logits_batch[bcnt]
        rts = rts_batch[bcnt]

        output_tokens_list = tokens.view(-1).contiguous()
        original_context = tokenizer.DecodeIds(output_tokens_list.tolist())
        context_length = len(tokens[0])
        logits = logits[0, -1]

        mct_tree_batch[bcnt] = []
        mct_tree_batch[bcnt].append([logits, rts, tokens, -np.ones(len(logits)), torch.ones(len(logits)).to(device), 0])
        original_context_batch.append(original_context)
        context_length_batch.append(context_length)

    final_result_batch = [[] for i in range(batch_num)]
    break_cond_batch = [False for i in range(batch_num)]

    nextid_batch = [0 for i in range(batch_num)]
    tries_batch = [0 for i in range(batch_num)]

    while True:
        for bcnt in range(batch_num):
            if break_cond_batch[bcnt]:
                continue
            if len(final_result_batch[bcnt]) >= num_candidates:
                break_cond_batch[bcnt] = True
            if tries_batch[bcnt] >= max_tries:
                break_cond_batch[bcnt] = True

        # print("break_cond_batch:", max_tries, break_cond_batch)
        # print("tries_batch:", tries_batch)

        all_break = sum(break_cond_batch)
        if all_break == batch_num:
            # print("all_break condition astisfied:", tries)
            break

        for bcnt in range(batch_num):
            if break_cond_batch[bcnt]:
                if mct_tree_batch.get(bcnt):
                    del mct_tree_batch[bcnt]
                    torch.cuda.empty_cache()

        # data of each branch
        input_ids_batch = {}
        position_ids_batch = {}
        attension_mask_batch = {}
        mems_batch = {}
        tokens_batch = {}
        tmp_batch = {}
        score_batch = {}

        for bcnt in range(batch_num):
            if break_cond_batch[bcnt]:
                continue

            mct_tree = mct_tree_batch[bcnt]
            final_result = final_result_batch[bcnt]
            original_context = original_context_batch[bcnt]
            context_length = context_length_batch[bcnt]
            nextid = nextid_batch[bcnt]
            length = length_batch[bcnt]
            tries = tries_batch[bcnt]

            ret_datas = get_sentence_postproc_model_input(
                tokenizer, args, device, endnote, max_tries, mct_tree, final_result, original_context, context_length, nextid, length, num_candidates, tries)
            if ret_datas is None:
                # print("final_result:", len(final_result))
                break_cond_batch[bcnt] = True
                continue
            my_input_ids, my_position_ids, my_attension_mask, my_mems, tokens, nextid, tmp, score, tries = ret_datas

            input_ids_batch[bcnt] = my_input_ids
            position_ids_batch[bcnt] = my_position_ids
            attension_mask_batch[bcnt] = my_attension_mask
            mems_batch[bcnt] = my_mems

            tokens_batch[bcnt] = tokens
            tmp_batch[bcnt] = tmp
            score_batch[bcnt] = score
            nextid_batch[bcnt] = nextid
            tries_batch[bcnt] = tries

        time1 = time.time()
        logits_batch_ret, rts_batch_ret = infer_model_by_batchs(model, input_ids_batch, position_ids_batch, attension_mask_batch, mems_batch, break_cond_batch)
        time2 = time.time()
        print("infer_model_by_batchs time:", time2 - time1)

        for bcnt in range(batch_num):
            if break_cond_batch[bcnt]:
                continue

            logits = logits_batch_ret[bcnt]
            rts  = rts_batch_ret[bcnt]
            tokens = tokens_batch[bcnt]
            tmp = tmp_batch[bcnt]
            score = score_batch[bcnt]
            mct_tree = mct_tree_batch[bcnt]

            logits = logits[0, -1] / tmp
            mct_tree.append([logits, rts, tokens, -np.ones(len(logits)), torch.ones(len(logits)).cuda(), score])
            nextid_batch[bcnt] = len(mct_tree) - 1

        input_ids_batch.clear()
        position_ids_batch.clear()
        attension_mask_batch.clear()
        mems_batch.clear()
        tokens_batch.clear()
        tmp_batch.clear()
        score_batch.clear()
        torch.cuda.empty_cache()

    mct_tree_batch.clear()
    torch.cuda.empty_cache()

    return final_result_batch


def infer_model_by_batchs(model, input_ids_batch, position_ids_batch, attension_mask_batch, mems_batch, break_cond_batch):
    class BatchModelData:
        def __init__(self):
            self.input_ids_batch = []
            self.position_ids_batch = []
            self.attension_mask_batch = []
            self.mems_batch = []
            self.idxs = [] # idx in batch to get order of data

    batch_num = len(break_cond_batch)
    # print("infer_model_by_batchs batch_num:", batch_num)
    batch_data_dict = {}

    for bcnt in range(batch_num):
        if break_cond_batch[bcnt]:
            continue

        input_id = input_ids_batch[bcnt]
        position_id = position_ids_batch[bcnt]
        attension_mas = attension_mask_batch[bcnt]
        mems = mems_batch[bcnt]
        shape_str = str(mems[0].shape)

        if batch_data_dict.get(shape_str) is None:
            batch_data_dict[shape_str] = BatchModelData()

        batch_data_dict[shape_str].input_ids_batch.append(input_id)
        batch_data_dict[shape_str].position_ids_batch.append(position_id)
        batch_data_dict[shape_str].attension_mask_batch.append(attension_mas)
        batch_data_dict[shape_str].mems_batch.append(mems)
        batch_data_dict[shape_str].idxs.append(bcnt)

    logits_batch_dict = {}
    rts_batch_dict = {}

    group_num = len(batch_data_dict)
    # print("infer_model_by_batchs batch num:", group_num)
    for shape_str in batch_data_dict:
        batch_data = batch_data_dict[shape_str]

        # time1 = time.time()
        input_ids = cat_tensors(batch_data.input_ids_batch)
        position_ids = cat_tensors(batch_data.position_ids_batch)
        attension_mask = cat_tensors(batch_data.attension_mask_batch)
        mems = get_mems_batch(batch_data.mems_batch)
        # time2 = time.time()
        # print("infer_model_by_batchs cat:", time2 -time1)

        # time1 = time.time()
        logits_batch, *rts_batch = model(input_ids, position_ids, attension_mask, *mems)
        # time2 = time.time()
        # print("infer_model_by_batchs model:", time2 -time1, input_ids.shape, mems[0].shape, shape_str)

        # time1 = time.time()
        logits_batch_n = torch.split(logits_batch, 1)
        rts_batch_n = split_mems_batch(rts_batch)
        # time2 = time.time()
        # print("infer_model_by_batchs split:", time2 -time1)

        del logits_batch
        rts_batch.clear()

        mini_batch_num = len(batch_data.idxs)
        for i in range(mini_batch_num):
            idx = batch_data.idxs[i]
            logits_batch_dict[idx] = logits_batch_n[i]
            rts_batch_dict[idx] = rts_batch_n[i]

    logits_ret = []
    rts_ret = []
    for bcnt in range(batch_num):
        logits_ret.append(logits_batch_dict.get(bcnt))
        rts_ret.append(rts_batch_dict.get(bcnt))

    batch_data_dict.clear()
    logits_batch_dict.clear()
    rts_batch_dict.clear()
    torch.cuda.empty_cache()
    return logits_ret, rts_ret


def get_sentence_postproc_model_input(tokenizer, args, device, endnote, max_tries, mct_tree, final_result, original_context, context_length, nextid, sentence_len, num_candidates, tries):
    min_length = sentence_len
    max_length = sentence_len
    curvote = get_curvote(endnote)

    while (len(final_result) < num_candidates) and (tries < max_tries):
        tries += 1
        currentid = nextid
        while currentid != -1:
            tc = torch.log(mct_tree[currentid][4])
            tc = tc + F.relu(tc - 10) * 1000
            logits = mct_tree[currentid][0].view(-1) - tc * 0.5
            logits = logits[:50001]
            log_probs = F.softmax(logits, dim=-1)

            pr = torch.multinomial(log_probs, num_samples=1)[0]
            # pr=torch.argmax(logits)
            prev = pr.item()
            # print(logits.shape,currentid,prev)
            mct_tree[currentid][4][prev] += 1
            lastid = currentid
            currentid = int(mct_tree[currentid][3][prev])
        # start from lastid & currentid

        cqs = mct_tree[lastid][2]

        tokens = torch.cat((cqs, pr.unsqueeze(0).view(1, 1)), dim=1)
        output_tokens_list = tokens.view(-1).contiguous()

        sentence = tokenizer.DecodeIds(output_tokens_list[context_length:].tolist())

        logit = mct_tree[lastid][0]
        log_probs = F.softmax(logit, dim=-1)
        log_pbs = torch.log(log_probs)
        score = log_pbs[prev].item()

        nextid = 0
        ip = checksentence(sentence, original_context, min_length, max_length, endnote, curvote=curvote)
        for j in final_result:
            if j[0] == sentence:
                ip = 1
            if ('<|end' in sentence) and ('<|end' in j[0]):
                ip = 1

        score = mct_tree[lastid][5] + score
        if (ip == 1):
            mct_tree[lastid][4][prev] = 10000
            continue
        if (ip == 0):
            mct_tree[lastid][4][prev] = 10000
            final_result.append([copy.deepcopy(sentence), copy.deepcopy(score), copy.deepcopy(tokens), copy.deepcopy(mct_tree[lastid][1])])
            continue

        mct_tree[lastid][3][prev] = len(mct_tree)
        tmp = args.temperature
        if (len(sentence) >= 4 or (len(sentence) == 3 and max_length == 5)):
            tmp = tmp * 0.6
        rts = mct_tree[lastid][1]
        index = len(tokens[0])
        my_input_ids = tokens[:, index - 1: index]
        my_position_ids = tokens.new_ones((1, 1)) * (index - 1)
        my_attension_mask = tokens.new_ones(1, 1, 1, args.mem_length + 1, device=tokens.device, dtype=torch.float)
        my_mems = rts

        return my_input_ids, my_position_ids, my_attension_mask, my_mems, tokens, nextid, tmp, score, tries
    return None


def generate_sentence(model, tokenizer, args, device, current_tokens, mems, endnote=[",", "，", "?", "？"], num_candidates=10, min_length=5, max_length=7):
    if min_length != max_length:
        mems = []
        tokens, attention_mask, position_ids = get_batch(current_tokens, device, args)
        logits, *rts = model(tokens, position_ids, attention_mask, *mems)
    else:
        tokens = current_tokens
        index = len(tokens[0])
        in_tokens = tokens[:, index - 1: index]
        in_position_ids = tokens.new_ones((1, 1)) * (index - 1)
        in_attention_mask = tokens.new_ones(1, 1, 1, args.mem_length + 1, device=tokens.device, dtype=torch.float)

        logits, *rts = model(in_tokens, in_position_ids, in_attention_mask, *mems)

    mct_tree = []
    output_tokens_list = tokens.view(-1).contiguous()
    original_context = tokenizer.DecodeIds(output_tokens_list.tolist())
    context_length = len(tokens[0])
    logits = logits[0, -1]

    mct_tree.append([logits, rts, tokens, -np.ones(len(logits)), torch.ones(len(logits)).to(device), 0])

    final_result = []
    nextid = 0
    max_tries = num_candidates * 30
    curvote = get_curvote(endnote)

    tries = 0
    while (len(final_result) < num_candidates) and (tries < max_tries):
        tries += 1
        currentid = nextid
        while currentid != -1:
            tc = torch.log(mct_tree[currentid][4])
            tc = tc + F.relu(tc - 10) * 1000
            logits = mct_tree[currentid][0].view(-1) - tc * 0.5
            logits = logits[:50001]
            log_probs = F.softmax(logits, dim=-1)

            pr = torch.multinomial(log_probs, num_samples=1)[0]
            # pr=torch.argmax(logits)
            prev = pr.item()
            # print(logits.shape,currentid,prev)
            mct_tree[currentid][4][prev] += 1
            lastid = currentid
            currentid = int(mct_tree[currentid][3][prev])
        # start from lastid & currentid

        cqs = mct_tree[lastid][2]

        tokens = torch.cat((cqs, pr.unsqueeze(0).view(1, 1)), dim=1)
        output_tokens_list = tokens.view(-1).contiguous()

        sentence = tokenizer.DecodeIds(output_tokens_list[context_length:].tolist())

        logit = mct_tree[lastid][0]
        log_probs = F.softmax(logit, dim=-1)
        log_pbs = torch.log(log_probs)
        score = log_pbs[prev].item()

        nextid = 0
        ip = checksentence(sentence, original_context, min_length, max_length, endnote, curvote=curvote)
        for j in final_result:
            if j[0] == sentence:
                ip = 1
            if ('<|end' in sentence) and ('<|end' in j[0]):
                ip = 1

        score = mct_tree[lastid][5] + score
        if (ip == 1):
            mct_tree[lastid][4][prev] = 10000
            continue
        if (ip == 0):
            mct_tree[lastid][4][prev] = 10000
            final_result.append([copy.deepcopy(sentence), copy.deepcopy(score), copy.deepcopy(tokens), copy.deepcopy(mct_tree[lastid][1])])
            continue

        mct_tree[lastid][3][prev] = len(mct_tree)
        tmp = args.temperature
        if (len(sentence) >= 4 or (len(sentence) == 3 and max_length == 5)):
            tmp = tmp * 0.6
        rts = mct_tree[lastid][1]
        index = len(tokens[0])
        my_input_ids = tokens[:, index - 1: index]
        my_position_ids = tokens.new_ones((1, 1)) * (index - 1)
        my_attension_mask = tokens.new_ones(1, 1, 1, args.mem_length + 1, device=tokens.device, dtype=torch.float)
        my_mems = rts

        logits, *rts = model(my_input_ids, my_position_ids, my_attension_mask, *my_mems)

        logits = logits[0, -1] / tmp

        mct_tree.append([logits, rts, tokens, -np.ones(len(logits)), torch.ones(len(logits)).cuda(), score])
        nextid = len(mct_tree) - 1

    del mct_tree
    torch.cuda.empty_cache()

    return final_result


def get_score_after_sentence_batch(model, tokenizer, args, device, i, id, w, title, 
                                   current_sentence, input_len, ini_score, best_score, best_pos, 
                                   gen, final_storage, beam_sentences, final_storage_score, past_beam_id, beam_new_sentences, overall_score):
    '''
    inplace update input data
    '''
    beam_sentences_id0 = beam_sentences[id][0]
    beam_sentences_id1 = beam_sentences[id][1]
    
    # print("gen score num after generate_sentence:", len(gen))

    def calc_score(score1, ss1, beam_sentences_id0):
        iscore = score1 - ss1
        if i >= 1:
            imp = 1
            if i % 2 == 0:
                imp += 1.5
            scorem = check2com(jj[0], beam_sentences_id0, imp)
            iscore += scorem
        return iscore

    input_str_batch = []
    output_str_batch = []
    ss1_batch = []
    gen_idxs = []

    for gen_idx, jj in enumerate(gen):
        if '<|end' in jj[0]:
            if (i % 2 == 1 and i >= 3):
                final_storage.append(copy.deepcopy(current_sentence[input_len:]))
                sc = beam_sentences_id1 / (i + 1)  # prioritize short poems
                sc = sc.item()
                if (i == 5 or i == 9 or i == 13):
                    sc -= 1.5
                if (i == 15):
                    sc -= 0.6
                if (i == 11):
                    sc -= 0.4
                if (i == 3):
                    sc += 0.2
                if sc > best_score:
                    best_score = sc
                    best_pos = len(final_storage) - 1
                sc = np.abs(sc)
                final_storage_score.append(sc)
                print("get_score_after_sentence_batch current_sentence:", current_sentence, final_storage_score[-1])
            continue
        st = jj[0]
        # experiment shows that this is better universal,
        if (i % 2 == 0):
            st = getlastsentence(beam_sentences_id0) + jj[0]
        else:
            st = get2sentencebefore(beam_sentences_id0) + ',' + getlastsentence(beam_sentences_id0) + jj[0]

        input_str = '”' + st + '”此句出自'
        output_str = "古诗《" + title + '》'
        
        past_beam_id.append(w)

        ss = -jj[1] / len(jj[0]) - 8
        # ss = ss.cpu().numpy()
        ss1 = 0.45 * (np.abs(ss) + ss)

        input_str_batch.append(input_str)
        output_str_batch.append(output_str)
        ss1_batch.append(ss1)
        gen_idxs.append(gen_idx)
        
    scores = generate_score_batch(model, tokenizer, args, device, input_str_batch, output_str_batch)

    for bcnt in range(len(scores)):
        jj = gen[gen_idxs[bcnt]]

        iscore = calc_score(scores[bcnt], ss1_batch[bcnt], beam_sentences_id0)
        jj[0] = beam_sentences_id0 + jj[0]
        jj[1] = iscore + ini_score

        beam_new_sentences.append(jj)
        overall_score.append(jj[1])

    return best_score, best_pos


def get_score_after_sentence(model, tokenizer, args, device, i, id, w, title, 
                             current_sentence, input_len, ini_score, best_score, best_pos, 
                             gen, final_storage, beam_sentences, final_storage_score, past_beam_id, beam_new_sentences, overall_score):
    '''
    inplace update input data
    '''
    beam_sentences_id0 = beam_sentences[id][0]
    beam_sentences_id1 = beam_sentences[id][1]

    def calc_score(score1, ss1, beam_sentences_id0):
        iscore = score1 - ss1
        if i >= 1:
            imp = 1
            if i % 2 == 0:
                imp += 1.5
            scorem = check2com(jj[0], beam_sentences_id0, imp)
            iscore += scorem
        return iscore
    print("gen score num after generate_sentence:", len(gen))

    for jj in gen:
        if '<|end' in jj[0]:
            if (i % 2 == 1 and i >= 3):
                final_storage.append(copy.deepcopy(current_sentence[input_len:]))
                sc = beam_sentences_id1 / (i + 1)  # prioritize short poems
                sc = sc.item()
                if (i == 5 or i == 9 or i == 13):
                    sc -= 1.5
                if (i == 15):
                    sc -= 0.6
                if (i == 11):
                    sc -= 0.4
                if (i == 3):
                    sc += 0.2
                if sc > best_score:
                    best_score = sc
                    best_pos = len(final_storage) - 1
                sc = np.abs(sc)
                final_storage_score.append(sc)
                print("get_score_after_sentence current_sentence:", current_sentence, final_storage_score[-1])
            continue
        st = jj[0]
        # experiment shows that this is better universal,
        if (i % 2 == 0):
            st = getlastsentence(beam_sentences_id0) + jj[0]
        else:
            st = get2sentencebefore(beam_sentences_id0) + ',' + getlastsentence(beam_sentences_id0) + jj[0]

        input_str = '”' + st + '”此句出自'
        output_str = "古诗《" + title + '》'
        
        past_beam_id.append(w)

        ss = -jj[1] / len(jj[0]) - 8
        ss1 = 0.45 * (np.abs(ss) + ss)

        print("generate_score after generate_sentence:", input_str, output_str)
        score1 = generate_score(model, tokenizer, args, device, input_str, output_str)

        iscore = calc_score(score1, ss1, beam_sentences_id0)
        jj[0] = beam_sentences_id0 + jj[0]
        jj[1] = iscore + ini_score

        beam_new_sentences.append(jj)
        overall_score.append(jj[1].cpu())

    return best_score, best_pos


def get_batch(context_tokens, device, args):
    # print("get_batch in0:", context_tokens.shape)
    tokens = context_tokens
    tokens = tokens.view(args.batch_size, -1).contiguous()
    tokens = tokens.to(device)
    # print("get_batch in1:", tokens.shape)

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_masks_and_position_ids(
        tokens,
        args.eod_token,
        reset_position_ids=False,
        reset_attention_mask=False,
        transformer_xl=args.transformer_xl,
        mem_length=args.mem_length)
    return tokens, attention_mask, position_ids


def get_penalty(mid_str, eval_str):
    penalty = 0
    title = mid_str[:-4]
    for i in eval_str:
        if i in title:
            penalty += 1
    return penalty


def build_mask_matrix(query_length, key_length, sep=0, device='cuda'):
    m = torch.ones((1, query_length, key_length), device=device)
    assert query_length <= key_length
    m[0, :, -query_length:] = torch.tril(m[0, :, -query_length:])
    m[0, :, :sep + (key_length - query_length)] = 1
    m = m.unsqueeze(1)
    return m


def get_gen_score_data(tokenizer, args, device, mid_str, eval_str):
    # penalty on same word
    mid_tokens = tokenizer.EncodeAsIds(mid_str).tokenization
    eval_tokens = tokenizer.EncodeAsIds(eval_str).tokenization
    context_tokens = mid_tokens

    context_tokens_tensor = torch.cuda.LongTensor(context_tokens + eval_tokens)
    tokens, attention_mask, position_ids = get_batch(context_tokens_tensor, device, args)

    index = 0
    tokens = tokens[:, index:]
    position_ids = torch.arange(index, tokens.shape[1], dtype=torch.long, device=tokens.device).unsqueeze(0)
    attention_mask = build_mask_matrix(tokens.shape[1] - index, tokens.shape[1], device=tokens.device)
    return tokens, position_ids, attention_mask, eval_tokens


def score_post_process(penalty, logits, eval_tokens):
    eval_length = len(eval_tokens)
    logits = logits[:, -eval_length - 1:-1]
    log_probs = F.softmax(logits, dim=-1)
    log_num = torch.log(log_probs).data.clamp(min=-35, max=100000)

    log_nums = [
        log_num[0, i, eval_token] for i, eval_token in enumerate(eval_tokens)  # TODO eos
    ]
    sumlognum = sum(log_nums).cpu().numpy()
    return sumlognum - 2.5 * (penalty**2.5)



def merge_batch_data(batch_datas):
    idxs = []
    tokens = []
    position_ids = []
    attention_masks = []

    for data in batch_datas:
        idxs.append(data.idx)
        tokens.append(data.tokens)
        position_ids.append(data.position_ids)
        attention_masks.append(data.attention_mask)

    batch_tokens = cat_tensors(tokens)
    batch_position_ids = cat_tensors(position_ids)
    batch_attention_masks = cat_tensors(attention_masks)

    return idxs, batch_tokens, batch_position_ids, batch_attention_masks

def generate_score_batch(model, tokenizer, args, device, mid_strs, eval_strs, raw_mems=None):
    class ModelData:
        def __init__(self, idx, tokens, position_ids, attention_mask):
            self.idx = idx
            self.tokens = tokens
            self.position_ids = position_ids
            self.attention_mask = attention_mask
            self.result = None

    data_num = len(mid_strs)
    penaltys = [get_penalty(mid_strs[i], eval_strs[i]) for i in range(data_num)]
    eval_tokens_all = []
    batch_data_dict = {}

    for i in range(data_num):
        _tokens, _position_ids, _attention_mask, eval_tokens = get_gen_score_data(tokenizer, args, device, mid_strs[i], eval_strs[i])
        shape_str = str(_tokens.shape)
        if batch_data_dict.get(shape_str) is None:
            batch_data_dict[shape_str] = []        
        in_data = ModelData(i, _tokens, _position_ids, _attention_mask)
        batch_data_dict[shape_str].append(in_data)
        eval_tokens_all.append(eval_tokens)

    # print("generate_score data_num:", data_num)
    # print("generate_score batch num:", len(batch_data_dict))

    scores = {}
    for shape_str in batch_data_dict:
        mems = []
        batch_datas = batch_data_dict[shape_str]
        idxs, batch_tokens, batch_position_ids, batch_attention_masks = merge_batch_data(batch_datas)
        logits, *_ = model(batch_tokens, batch_position_ids, batch_attention_masks, *mems)

        # print("generate_score batch out logits:", logits.shape, batch_tokens.shape)
        
        logits_batch = torch.split(logits, 1)
        for i in range(len(idxs)):
            g_idx = idxs[i]
            score = score_post_process(penaltys[g_idx], logits_batch[i], eval_tokens_all[g_idx])
            scores[g_idx] = score

    scores_all = []
    for i in range(len(scores)):
        scores_all.append(scores[i])
    # del logits
    # del mems
    # torch.cuda.empty_cache()
    # print("generate_score batch out scores:", scores)
    # print("generate_score batch out scores_all:", scores_all)

    return scores_all


def generate_score(model, tokenizer, args, device, mid_str, eval_str, raw_mems=None):
    penalty = get_penalty(mid_str, eval_str)

    # penalty on same word
    mid_tokens = tokenizer.EncodeAsIds(mid_str).tokenization
    eval_tokens = tokenizer.EncodeAsIds(eval_str).tokenization
    context_tokens = mid_tokens

    context_length = len(context_tokens)
    eval_length = len(eval_tokens)

    context_tokens_tensor = torch.cuda.LongTensor(context_tokens + eval_tokens)
    # print("generate_score context_tokens_tensor:", context_tokens_tensor.shape)

    tokens, attention_mask, position_ids = get_batch(context_tokens_tensor, device, args)

    index = 0

    tokens = tokens[:, index:]
    position_ids = torch.arange(index, tokens.shape[1], dtype=torch.long, device=tokens.device).unsqueeze(0)
    attention_mask = build_mask_matrix(tokens.shape[1] - index, tokens.shape[1], device=tokens.device)
    mems = []

    # print("generate_score in:", tokens.shape, position_ids.shape, attention_mask.shape)

    logits, *mems = model(tokens, position_ids, attention_mask, *mems)

    logits = logits[:, -eval_length - 1:-1]
    log_probs = F.softmax(logits, dim=-1)
    log_num = torch.log(log_probs).data.clamp(min=-35, max=100000)

    log_nums = [
        log_num[0, i, eval_token] for i, eval_token in enumerate(eval_tokens)  # TODO eos
    ]

    sumlognum = sum(log_nums)

    del logits
    del mems
    torch.cuda.empty_cache()

    return sumlognum - 2.5 * (penalty**2.5)


def generate_token_tensor(str, tokenizer):
    with torch.no_grad():
        context_tokens = tokenizer.EncodeAsIds(str).tokenization
        context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
        return context_tokens_tensor


def checkpz(st, wd):
    rus = set(['八', '搭', '塌', '邋', '插', '察', '杀', '煞', '夹', '俠', '瞎', '辖', '狹', '匣', '黠', '鸭', '押', '压', '刷', '刮', '滑', '猾', '挖', '蜇', '舌', '鸽', '割', '胳', '搁', '瞌', '喝', '合', '盒', '盍', '曷', '貉', '涸', '劾', '核', '钵', '剝', '泼', '摸', '脱', '托', '捋', '撮', '缩', '豁', '活', '切', '噎', '汁', '织', '隻', '掷', '湿', '虱', '失', '十', '什', '拾', '实', '食', '蝕', '识', '石', '劈', '霹', '滴', '踢',
               '剔', '屐', '积', '激', '击', '漆', '吸', '息', '媳', '昔', '席', '锡', '檄', '觋', '揖', '一', '壹', '扑', '匍', '仆', '弗', '紱', '拂', '福', '蝠', '幅', '辐', '服', '伏', '茯', '督', '突', '秃', '俗', '出', '蜀', '窟', '哭', '忽', '惚', '斛', '鹄', '屋', '屈', '诎', '曲', '戌', '拍', '塞', '摘', '拆', '黑', '勺', '芍', '嚼', '粥', '妯', '熟', '白', '柏', '伯', '薄', '剥', '摸', '粥', '轴', '舳', '妯', '熟', '角', '削', '学'])
    ss = set(['de', 'te', 'le', 'ze', 'ce', 'se', 'fa', 'fo', 'dei', 'zei', 'gei', 'hei', 'sei',
              'bie', 'pie', 'mie', 'die', 'tie', 'nie', 'lie', 'kuo', 'zhuo', 'chuo', 'shuo', 'ruo'])
    # 入声字判断

    # 轻声按失败算。
    if not(st[-1] in ['1', '2', '3', '4']):
        return 0

    if open_old_pronounce == 1:
        if wd in rus:
            return 2
        if wd in ['嗟', '瘸', '靴', '爹']:
            return 1
        if st[:-1] in ss:
            return 2

        if (st[-1] == '2' and st[0] in ['b', 'd', 'g', 'j', 'z']):
            return 2
        if 'ue' in st:
            return 2

    if st[-1] in ['1', '2']:
        return 1
    return 2


# inner rhy, must obey
def checkrhyself(sentence):
    if len(sentence) == 0:
        return 0
    st = sentence
    fullst = False
    while (len(st) > 0 and st[-1] in [',', '。', '，', '?', '？', '!', '！']):
        st = st[:-1]
        fullst = True

    l1 = pinyin(st, style=TONE3)
    if len(l1) < len(st):
        print(l1, sentence)
        return 1
    for i in l1:
        if len(i[0]) < 2:
            return 1
    if len(st) <= 3:
        return 2

    pz1 = checkpz(l1[1][0], sentence[1])

    if len(st) >= 4:
        if len(l1[3]) < 1:
            print(sentence, l1)
        pz2 = checkpz(l1[3][0], sentence[3])
        if pz2 + pz1 != 3:
            return 1
    if len(st) >= 6:
        if len(l1[5]) < 1:
            print(sentence, l1)
        pz3 = checkpz(l1[5][0], sentence[5])
        if pz2 + pz3 != 3:
            return 1
    if fullst:
        if len(sentence) < 6:
            return 1
        pz11 = checkpz(l1[-3][0], st[-3])
        pz12 = checkpz(l1[-2][0], st[-2])
        pz13 = checkpz(l1[-1][0], st[-1])
        if (pz11 == pz12) and (pz12 == pz13):
            return 1
    return 2


def checkrhy(sentence, last, imp, req=0):

    while (len(sentence) > 0 and (sentence[-1] in [',', '。', '，', '?', '？', '!', '！'])):
        sentence = sentence[:-1]
    if len(sentence) == 0:
        return 0

    while last[-1] in [',', '。', '，', '?', '？', '!', '！']:
        last = last[:-1]
    l1 = pinyin(sentence, style=TONE3)
    l2 = pinyin(last, style=TONE3)
    # print(l1,l2)
    disobey = 0
    if len(l1) != len(sentence):
        return -1000
    for i in range(len(sentence)):
        if (i < len(l1)) and (i < len(l2)):
            st1 = checkpz(l1[i][0], sentence[i])

            sr1 = checkpz(l2[i][0], last[i])
            if (req == 1 and i % 2 == 1):
                st1 = 3 - st1

            if st1 + sr1 != 3:
                if req == 0:
                    disobey += 0.35
                if i % 2 == 1:
                    disobey += 0.35
                    if req == 1:
                        disobey += 0.2
                if i == len(l2) - 1:
                    disobey += 0.65
                    if req == 1:
                        disobey += 0.35

    disobey *= imp
    disobey = -5 * disobey / len(l2)
    for i in range(len(l1)):
        for j in range(i + 2, len(l1)):
            if l1[i][0][:-1] == l1[j][0][:-1]:
                disobey -= 7 / len(l1)
    return disobey


def checksentence(sentence, original_context, min_length, max_length, endnote, curvote=0):
    if "<|end" in sentence:
        return 0

    if "的" in sentence:
        return 1
    if len(sentence) == 0:
        return 1
    if ((len(sentence) > max_length and not(sentence[-1] in endnote)) or len(sentence) == 0) or len(sentence) > max_length + 1:
        return 1
    if (sentence[-1] in endnote) and ((len(sentence) <= min_length) or (len(sentence) == 7)):
        return 1

    if (sentence[-1] in endnote) and (sentence[:-1] in original_context):
        return 1
    last = getlastsentence(original_context)

    mdisobey = 0
    illegal_notes = [' ', ':', '《', '》', '‘', '“', '-', '——', '⁇',
                     '[', '【', '】', ']', '.', '、', '(', '（', ')', '）', '·']
    if '。' in endnote:
        illegal_notes.extend([',', '，'])
    else:
        illegal_notes.append('。')
    for i in range(10):
        illegal_notes.append(str(i))
    for i in range(64, 123):
        illegal_notes.append(chr(i))
    for note in illegal_notes:
        if note in sentence:
            return 1
    if min_length == max_length:
        imp = 1
        if (',' in last) or('，' in last):
            imp = 1.5

        if curvote == 0:
            rt = checkrhy(sentence, last, imp, req=1)
        else:
            rt = checkrhy(sentence, last, imp)
        if rt < -0.75:
            return 1

    for i in range(len(sentence)):
       # if sentence[i]=="柯":
        #    print(sentence[i],last[i],sentence[i]==last[i])
        if min_length == max_length:
            if (i < len(last) - 1) and (sentence[i] == last[i]):
                # print(sentence,last)
                return 1

        if i < len(sentence) - 3:
            if sentence[i:i + 3] in original_context:
                return 1
            if sentence[i:i + 2] in sentence[:i]:
                return 1

    if checkrhyself(sentence) == 1:
        return 1
    if (sentence[-1] in endnote):
        return 0
    return 2


def getlength(str):
    w = str.replace('。', ',').replace('，', ',').replace('？', ',').replace('?', ',').replace(
        ' ', ',').replace('！', ',').replace('!', ',').replace(':', ',').replace(' ', '')
    sp = w.split(',')
    return len(sp[-2])


def getlastsentence(str):
    w = str.replace('。', ',').replace('，', ',').replace('？', ',').replace('?', ',').replace(
        ' ', ',').replace('！', ',').replace('!', ',').replace(':', ',').replace(' ', '')
    sp = w.split(',')
    fom = sp[-1]
    if len(fom) == 0:
        fom = sp[-2]
    return fom + str[-1]


def get2sentencebefore(str):
    w = str.replace('。', ',').replace('，', ',').replace('？', ',').replace('?', ',').replace(
        ' ', ',').replace('！', ',').replace('!', ',').replace(':', ',').replace(' ', '')
    sp = w.split(',')
    idk = -1
    while len(sp[idk]) == 0:
        idk -= 1
    idk -= 1
    while len(sp[idk]) == 0:
        idk -= 1
    return sp[idk]


def check2compare(sentence1, sentence2, imp):
    s1 = sentence1.replace('。', '').replace('，', '').replace('？', '').replace(
        '?', '').replace('  ', '').replace('！', '').replace('!', '').replace(',', '')
    s2 = sentence2.replace('。', '').replace('，', '').replace('？', '').replace(
        '?', '').replace(' ', '').replace('！', '').replace('!', '').replace(',', '')
    if len(s1) != len(s2):
        return -1000
    num = 0
    for i in range(len(s1)):
        if s1[i] == s2[i]:
            num += 1

    score = 0.5 - num * num * 2.5

    w1 = pinyin(s1, style=FINALS)[-1][0]
    w2 = pinyin(s2, style=FINALS)[-1][0]
    w3 = pinyin(s1)[-1]
    w4 = pinyin(s2)[-1]
    if (w1 != w2) or (s1[-1] == s2[-1]):
        score -= imp * 0.6
    group = [['a', 'ia', 'ua'], ['ai', 'uai', 'ei', 'ui', 'uei'], ['an', 'uan', 'ian', 'ie', 'ue', 've'],
             ['ou', 'iu', 'iou'], ['ang', 'iang', 'uang'], ['ao', 'iao'], ['e', 'o', 'uo'], ['en', 'un', 'uen', 'ong', 'iong', 'in', 'ing', 'er']]
    if (w1 != w2) and (s1[-1] != s2[-1]):
        for i in group:
            if (w1 in i) and (w2 in i):
                score += imp * 1
    if (w1 == w2) and (w3 != w4):
        score += imp * 1
    return score


def check2com(sentence, org_context, imp):

    before2 = get2sentencebefore(org_context)
    before1 = getlastsentence(org_context)[:-1]
    s1 = check2compare(sentence, before2, imp)
    if imp == 1:
        s2 = checkrhy(sentence, before1, imp + 0.5, req=1)
    else:
        s2 = checkrhy(sentence, before1, imp)
    sc = s1 + s2 + imp
    for i in range(len(sentence) - 1):
        if sentence[i] in org_context:
            sc -= 3
            if sentence[i:i + 2] in org_context:
                sc -= 5
                if (',' in sentence[i:i + 2]) or ('，' in sentence[i:i + 2]) or ('。' in sentence[i:i + 2]) or ('？' in sentence[i:i + 2]) or('?' in sentence[i:i + 2]) or ('！' in sentence[i:i + 2]) or ('!' in sentence[i:i + 2]):
                    sc -= 35
    return sc


def prepare_tokenizer(args):
    tokenizer_args = {
        'tokenizer_type': args.tokenizer_type,
        'corpus': None,
        'model_path': args.tokenizer_path,
        'vocab_size': args.vocab_size,
        'model_type': args.tokenizer_model_type,
        'cache_dir': args.cache_dir}
    tokenizer = make_tokenizer(**tokenizer_args)

    num_tokens = tokenizer.num_tokens
    before = num_tokens
    after = before
    multiple = args.make_vocab_size_divisible_by * \
        mpu.get_model_parallel_world_size()
    while (after % multiple) != 0:
        after += 1
    print_rank_0('> padded vocab (size: {}) with {} dummy '
                 'tokens (new size: {})'.format(
                     before, after - before, after))

    args.tokenizer_num_tokens = after
    args.tokenizer_num_type_tokens = tokenizer.num_type_tokens
    args.eod_token = tokenizer.get_command('eos').Id

    # after = tokenizer.num_tokens
    # while after % mpu.get_model_parallel_world_size() != 0:
    #     after += 1

    args.vocab_size = after
    print("prepare tokenizer done", flush=True)
    return tokenizer


def set_args():
    print("set args")
    args = get_args()
    print(args.gpu)
    # os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    # set up
    # print(args)
    args.deepspeed = True
    args.num_nodes = 1
    args.num_gpus = 1
    args.model_parallel_size = 1
    args.deepspeed_config = "script_dir/ds_config.json"
    args.num_layers = 32
    args.hidden_size = 2560
    args.load = "/mnt/zhiyuan/poem/latest"
    # "/mnt3/ckp/checkpoint2/new"
    args.num_attention_heads = 32
    args.max_position_embeddings = 1024
    args.tokenizer_type = "ChineseSPTokenizer"
    args.cache_dir = "cache"
    args.fp16 = True
    args.out_seq_length = 180
    args.seq_length = 200
    args.mem_length = 256
    args.transformer_xl = True
    args.temperature = 1.25
    args.top_k = 0
    args.top_p = 0
    return args


def setup_model(args):
    """Setup model and optimizer."""

    model = get_model(args)
    print("xxxx args{}.".format(args))
    '''
    if args.deepspeed:
         print_rank_0("DeepSpeed is enabled.")
    
         model, _, _, _ = deepspeed.initialize(
             model=model,
             model_parameters=model.parameters(),
             args=args,
             mpu=mpu,
             dist_init_required=False
         )
    '''
    if args.load is not None:
        if args.deepspeed:
            iteration, release, success = get_checkpoint_iteration(args)
            print(iteration)
            path = os.path.join(args.load, str(iteration), "mp_rank_00_model_states.pt")
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint["module"])
            #model = torch.jit.script(model)
        else:
            _ = load_checkpoint(
                model, None, None, args, load_optimizer_states=False)

    # if args.deepspeed:
    #     model = model.module

    # print("one",next(model.parameters()).is_cuda)
    return model


def prepare_model():
    """Main training program."""

    print('Generate Samples')

    # Disable CuDNN.
    torch.backends.cudnn.enabled = True

    print("disable cudnn")
    # Timer.
    timers = Timers()
    print("set timmer finished")
    # Arguments.
    args = set_args()
    print("set_args")
    # print(args)
    args.mem_length = args.seq_length + args.mem_length - 1

    # Pytorch distributed.
    initialize_distributed(args)
    print("initialize_distributed")

    # Random seeds for reproducability.
    args.seed = random.randint(0, 1000000)
    set_random_seed(args.seed)
    print("prepare_tokenizer")
    # get the tokenizer
    tokenizer = prepare_tokenizer(args)
    print("setup models")
    # Model, optimizer, and learning rate.
    model = setup_model(args)
    model.eval()
    # args.load="../ckp/txl-2.8b11-20-15-10"
    # model2=setup_model(args)
    # setting default batch size to 1
    args.batch_size = 1
    # torch.save(model, "dumped_model.pth")

    # generate samples
    return model, tokenizer, args


if __name__ == "__main__":
    out = generate()
