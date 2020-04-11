#!/usr/bin/env python3
#coding=utf-8

#============================#
#Program:utils.py
#       
#Date:20-4-11
#Author:liheng
#Version:V1.0
#============================#

import yaml
import os

def get_config(cfg_file:str):
    with open(cfg_file,'r',encoding='utf-8') as cfg:
        content = cfg.read()
        return yaml.load(content)