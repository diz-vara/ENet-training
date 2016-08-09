----------------------------------------------------------------------
-- Cityscape data loader,
-- Abhishek Chaurasia,
-- February 2016
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- to visualize the dataset

require 'color2idx' -- converter from my colors to index

torch.setdefaulttensortype('torch.FloatTensor')
----------------------------------------------------------------------
-- Cityscape dataset:

local trsize, tesize

trsize = 50 -- cityscape train images
tesize = 8  -- cityscape validation images
local classes = {'Unlabeled', 'Road', 'Sidewalk', 'Building', 'Wall', 'Fence',
  'Pole', 'TrafficLight', 'TrafficSign', 'Vegetation', 'Terrain',
  'Sky', 'Person', 'Rider', 'Car', 'Truck',
  'Bus', 'Train', 'Motorcycle', 'Bicycle'}
local conClasses = {'Road', 'Sidewalk', 'Building', 'Wall', 'Fence',
  'Pole', 'TrafficLight', 'TrafficSign', 'Vegetation', 'Terrain',
  'Sky', 'Person', 'Rider','Car', 'Truck',
  'Bus', 'Train', 'Motorcycle', 'Bicycle'} -- 19 classes

local nClasses = #classes

--------------------------------------------------------------------------------

-- Ignoring unnecessary classes
print '==> remapping classes'
local classMap = {[-1] =  {1}, -- licence plate
  [0]  =  {1}, -- Unlabeled
  [1]  =  {1}, -- Ego vehicle
  [2]  =  {1}, -- Rectification border
  [3]  =  {1}, -- Out of roi
  [4]  =  {1}, -- Static
  [5]  =  {1}, -- Dynamic
  [6]  =  {1}, -- Ground
  [7]  =  {2}, -- Road
  [8]  =  {3}, -- Sidewalk
  [9]  =  {1}, -- Parking
  [10] =  {1}, -- Rail track
  [11] =  {4}, -- Building
  [12] =  {5}, -- Wall
  [13] =  {6}, -- Fence
  [14] =  {1}, -- Guard rail
  [15] =  {1}, -- Bridge
  [16] =  {1}, -- Tunnel
  [17] =  {7}, -- Pole
  [18] =  {1},  -- Polegroup
  [19] =  {8}, -- Traffic light
  [20] =  {9}, -- Traffic sign
  [21] = {10}, -- Vegetation
  [22] = {11}, -- Terrain
  [23] = {12}, -- Sky
  [24] = {13}, -- Person
  [25] = {14}, -- Rider
  [26] = {15}, -- Car
  [27] = {16}, -- Truck
  [28] = {17}, -- Bus
  [29] =  {1}, -- Caravan
  [30] =  {1}, -- Trailer
  [31] = {18}, -- Train
  [32] = {19}, -- Motorcycle
  [33] = {20}, -- Bicycle
}

-- Reassign the class numbers
local classCount = 1

if opt.smallNet then
  classMap = {
    [-1] = {0},  -- licence platete
    [0]  = {0},  -- Unlabeled
    [1]  = {1},  -- Ego vehicle
    [2]  = {2},  -- Rectification border
    [3]  = {3},  -- Out of roi
    [4]  = {4},  -- Static
    [5]  = {5},  -- Dynamic
    [6]  = {6},  -- Ground
    [7]  = {7},  -- Road
    [8]  = {8}  -- Sidewalk
  }

  classes = {'Unlabeled', 'flat', 'construction', 'object', 'nature',
    'sky', 'human', 'vehicle', 'ego'}
  conClasses = {'flat', 'construction', 'object', 'nature','sky', 'human', 'vehicle', 'ego'} -- 7 classee
end
-- From here #class will give number of classes even after shortening the list
-- nClasses should be used to get number of classes in original list

-- saving training histogram of classes
local histClasses = torch.Tensor(#classes):zero()

print('==> number of classes: ' .. #classes)
print('classes are:')
print(classes)

--------------------------------------------------------------------------------
print '==> loading CoCar dataset'
local trainData, testData
local loadedFromCache = false
paths.mkdir(paths.concat(opt.cachepath, 'cocar'))
local cocarCachePath = paths.concat(opt.cachepath, 'cocar', 'data.t7')

if opt.cachepath ~= "none" and paths.filep(cocarCachePath) then
  local dataCache = torch.load(cocarCachePath)
  trainData = dataCache.trainData
  testData = dataCache.testData
  histClasses = dataCache.histClasses
  loadedFromCache = true
  print ('-- loaded from cache : ' .. cocarCachePath)
  dataCache = nil
  collectgarbage()
else
  local function has_image_extensions(filename)
    local ext = string.lower(path.extension(filename))

    -- compare with list of image extensions
    local img_extensions = {'.jpeg', '.jpg', '.png', '.ppm', '.pgm'}
    for i = 1, #img_extensions do
      if ext == img_extensions[i] then
        return true
      end
    end
    return false
  end

  -- initialize data structures:
  trainData = {
    data = torch.FloatTensor(trsize, opt.channels, opt.imHeight, opt.imWidth),
    labels = torch.FloatTensor(trsize, opt.labelHeight, opt.labelWidth),
    preverror = 1e10, -- a really huge number
    size = function() return trsize end,
    names = {}
  }

  testData = {
    data = torch.FloatTensor(tesize, opt.channels, opt.imHeight, opt.imWidth),
    labels = torch.FloatTensor(tesize, opt.labelHeight, opt.labelWidth),
    preverror = 1e10, -- a really huge number
    size = function() return tesize end,
    names = {}
  }


  local dpathRoot = opt.datapath .. '/orig/train/'
  
  print('==> loading training files from ' .. dpathRoot);

  assert(paths.dirp(dpathRoot), 'No training folder found at: ' .. opt.datapath)
  --load training images and labels:
  local c = 1
  for dir in paths.iterdirs(dpathRoot) do
    local dpath = dpathRoot .. dir .. '/'
    for file in paths.iterfiles(dpath) do
      --print(file)
      -- process each image
      if has_image_extensions(file) and c <= trsize then
        local imgPath = path.join(dpath, file)

        --load training images:
        local dataTemp = image.load(imgPath)
        trainData.data[c] = image.scale(dataTemp,opt.imWidth, opt.imHeight)
        trainData.names[c] = imgPath

        -- Load training labels:
        -- Load labels with same filename as input image.
        local lblPath = string.gsub(imgPath, "orig", "lbl")
        --imgPath = string.gsub(imgPath, ".png", "_labelIds.png")


        -- label image data are resized to be [1,nClasses] in [0 255] scale:
        local labelIn = image.load(lblPath, 3)
        local cc = (10*(labelIn[1] + labelIn[2]*2 + labelIn[3]*4)):ceil()
        local labelFile = image.scale(cc, opt.labelWidth, opt.labelHeight, 'simple'):float()

        
        labelFile:apply(color2idx)
        --torch.save(file..'.a.t7a',labelFile,'ascii')

        --print(imgPath .. ' ; ' .. lblPath)
        -- Syntax: histc(data, bins, min, max)
        histClasses = histClasses + torch.histc(labelFile, #classes, 1, #classes)

        -- convert to int and write to data structure:
        trainData.labels[c] = labelFile

        c = c + 1
        if c % 20 == 0 then
          xlua.progress(c, trsize)
        end
        collectgarbage()
      end
    end
  end
  print('')

  print('==> loading testing files');
  dpathRoot = opt.datapath .. '/orig/val/'

  assert(paths.dirp(dpathRoot), 'No testing folder found at: ' .. opt.datapath)
  -- load test images and labels:
  local c = 1
  for dir in paths.iterdirs(dpathRoot) do
    local dpath = dpathRoot .. dir .. '/'
    for file in paths.iterfiles(dpath) do

      -- process each image
      if has_image_extensions(file) and c <= tesize then
        local imgPath = path.join(dpath, file)

        --load training images:
        local dataTemp = image.load(imgPath)
        testData.data[c] = image.scale(dataTemp, opt.imWidth, opt.imHeight)
        testData.names[c] = imgPath;

        -- Load validation labels:
        -- Load labels with same filename as input image.
        imgPath = string.gsub(imgPath, "orig", "lbl")
        -- imgPath = string.gsub(imgPath, ".png", "_labelIds.png")


        -- load test labels:
        -- label image data are resized to be [1,nClasses] in in [0 255] scale:
        local labelIn = image.load(imgPath, 3)
        local cc = (10*(labelIn[1] + labelIn[2]*2 + labelIn[3]*4)):ceil()
        local labelFile = image.scale(cc, opt.labelWidth, opt.labelHeight, 'simple'):float()

        labelFile:apply(color2idx)

        -- convert to int and write to data structure:
        testData.labels[c] = labelFile

        c = c + 1
        if c % 20 == 0 then
          xlua.progress(c, tesize)
        end
        collectgarbage()
      end
    end
  end
end

if opt.cachepath ~= "none" and not loadedFromCache then
  print('==> saving data to cache: ' .. cocarCachePath)
  local dataCache = {
    trainData = trainData,
    testData = testData,
    histClasses = histClasses
  }
  torch.save(cocarCachePath, dataCache)
  dataCache = nil
  collectgarbage()
end

----------------------------------------------------------------------
print '==> verify statistics'

-- It's always good practice to verify that data is properly
-- normalized.

for i = 1, opt.channels do
  local trainMean = trainData.data[{ {},i }]:mean()
  local trainStd = trainData.data[{ {},i }]:std()

  local testMean = testData.data[{ {},i }]:mean()
  local testStd = testData.data[{ {},i }]:std()

  print('training data, channel-'.. i ..', mean: ' .. trainMean)
  print('training data, channel-'.. i ..', standard deviation: ' .. trainStd)

  print('test data, channel-'.. i ..', mean: ' .. testMean)
  print('test data, channel-'.. i ..', standard deviation: ' .. testStd)
end

----------------------------------------------------------------------

local classes_td = {[1] = 'classes,targets\n'}
for _,cat in pairs(classes) do
  table.insert(classes_td, cat .. ',1\n')
end

local file = io.open(paths.concat(opt.save, 'categories.txt'), 'w')
file:write(table.concat(classes_td))
file:close()

-- Exports
opt.dataClasses = classes
opt.dataconClasses  = conClasses
opt.datahistClasses = histClasses

return {
  trainData = trainData,
  testData = testData,
  mean = trainMean,
  std = trainStd
}
