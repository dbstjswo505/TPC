from typing import Type
import torch
import os
import pdb
import numpy as np
import copy

def register_person(model, p):
    conv_module = model.unet.up_blocks[1].resnets[1]
    setattr(conv_module, 'p', p)
    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 'p', p)
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 'p', p)
    for res in down_res_dict:
        for block in down_res_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 'p', p)
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 'p', p)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, 'p', p)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn2
    setattr(module, 'p', p)

def register_cal(model, cal):
    conv_module = model.unet.up_blocks[1].resnets[1]
    setattr(conv_module, 'cal', cal)
    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 'cal', cal)
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 'cal', cal)
    for res in down_res_dict:
        for block in down_res_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 'cal', cal)
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 'cal', cal)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, 'cal', cal)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn2
    setattr(module, 'cal', cal)

def register_time(model, t):
    conv_module = model.unet.up_blocks[1].resnets[1]
    setattr(conv_module, 't', t)
    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 't', t)
    for res in down_res_dict:
        for block in down_res_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 't', t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, 't', t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn2
    setattr(module, 't', t)

def register_feed(model, fd):
    conv_module = model.unet.up_blocks[1].resnets[1]
    setattr(conv_module, 'fd', fd)
    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 'fd', fd)
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 'fd', fd)
    for res in down_res_dict:
        for block in down_res_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 'fd', fd)
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 'fd', fd)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, 'fd', fd)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn2
    setattr(module, 'fd', fd)

def register_mask(model, c_mask):
    conv_module = model.unet.up_blocks[1].resnets[1]
    setattr(conv_module, 'c_mask', c_mask)
    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 'c_mask', c_mask)
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 'c_mask', c_mask)
    for res in down_res_dict:
        for block in down_res_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 'c_mask', c_mask)
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 'c_mask', c_mask)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, 'c_mask', c_mask)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn2
    setattr(module, 'c_mask', c_mask)

def register_p_mask(model, p_mask):
    conv_module = model.unet.up_blocks[1].resnets[1]
    setattr(conv_module, 'p_mask', p_mask)
    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 'p_mask', p_mask)
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 'p_mask', p_mask)
    for res in down_res_dict:
        for block in down_res_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 'p_mask', p_mask)
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 'p_mask', p_mask)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, 'p_mask', p_mask)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn2
    setattr(module, 'p_mask', p_mask)
