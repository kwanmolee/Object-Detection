#!/bin/bash

python3 downloadOI.py --classes 'Castle,Couch,Christmas_tree,Taxi,Penguin,Cookie,Apple,Swimming_pool,Deer,Porch,Bread,Bowling_equipment,Television,Fountain,Lifejacket,Lamp,Fedora,Bed,Beetle,Pillow,Ski,Carnivore,Platter,Sheep,Elephant,Human_beard' --mode train

python3 downloadOI.py --classes 'Castle,Couch,Christmas_tree,Taxi,Penguin,Cookie,Apple,Swimming_pool,Deer,Porch,Bread,Bowling_equipment,Television,Fountain,Lifejacket,Lamp,Fedora,Bed,Beetle,Pillow,Ski,Carnivore,Platter,Sheep,Elephant,Human_beard' --mode validation

python3 downloadOI.py --classes 'Castle,Couch,Christmas_tree,Taxi,Penguin,Cookie,Apple,Swimming_pool,Deer,Porch,Bread,Bowling_equipment,Television,Fountain,Lifejacket,Lamp,Fedora,Bed,Beetle,Pillow,Ski,Carnivore,Platter,Sheep,Elephant,Human_beard' --mode test