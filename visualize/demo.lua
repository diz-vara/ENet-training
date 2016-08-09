#!/usr/bin/env qlua
--------------------------------------------------------------------------------
-- View demos based on the trained network output
--
-- e-Lab
-- Written by: Abhishek Chaurasia
-- Dated: 24th March, 2016
--------------------------------------------------------------------------------

-- Torch packages
require 'image'
require 'imgraph'
require 'qtwidget'
require 'cunn'
require 'cudnn'


print (arg)

-- Local repo files
local opts = require 'opts'
-- Get the input arguments parsed and stored in opt
local opt = opts.parse(arg)
torch.save('otps.t7a',opt, 'ascii')
local colorMap = assert(require('colorMap'))

print (opt)

torch.setdefaulttensortype('torch.FloatTensor')
if opt.dev:lower() == 'cuda' then
   cutorch.setDevice(opt.devID)
   print("GPU # " .. cutorch.getDevice() .. " selected")
end

----------------------------------------
-- Network
local network = {}
network.path = opt.modelpath
if network.path == nil then
   network.path=opt.dmodel .. opt.model .. '/model-' .. opt.net .. '.net'
   network.txtpath=opt.dmodel .. opt.model .. '/model-' .. opt.net .. '.n7a'
else
  local rn = paths.dirname(network.path)
  opt.model = paths.basename(rn)
  opt.dmodel = paths.dirname(rn) .. '/'
end
assert(paths.filep(network.path), 'Model not present at ' .. network.path)
print("Loading model from: " .. network.path)

network.model = torch.load(network.path)

-- Convert all the modules in nn from cudnn
if #network.model:findModules('cudnn.SpatialConvolution') > 0 then
   if network.model.__typename == 'nn.DataParallelTable' then
      network.model = network.model:get(1)
   end
end

-- Change model type based on device being used for demonstration
if opt.dev:lower() == 'cpu' then
   cudnn.convert(network.model, nn)
   network.model:float()
   --torch.save(network.txtpath, network.model, 'ascii')
else
   network.model:cuda()
end

-- Set the module mode 'train = false'
network.model:evaluate()
network.model:clearState()

-- Get mean and std of the dataset used while training
local stat_file = opt.dmodel .. opt.model .. '/' .. 'stat.t7'
if paths.filep(stat_file) then
   network.stat = torch.load(stat_file)
elseif paths.filep(stat_file .. 'ascii')
then
   network.stat = torch.load(stat_file .. '.ascii', 'ascii')
else
   print('No stat file found in directory: ' .. opt.dmodel .. opt.model)
   network.stat = {}
   network.stat.mean = torch.Tensor{0, 0, 0}
   network.stat.std = torch.Tensor{1, 1, 1}
end

-- classes and color based on neural net model used:
local classes

--change target based on categories csv file:
function readCatCSV(filepath)
   print(filepath)
   local file = io.open(filepath, 'r')
   local classes = {}
   local targets = {}
   file:read()    -- throw away first line
   local fline = file:read()
   while fline ~= nil do
      local col1, col2 = fline:match("([^,]+),([^,]+)")
      table.insert(classes, col1)
      table.insert(targets, ('1' == col2))
      fline = file:read()
   end
   return classes, targets
end

-- Load categories from the list of categories generated during training
local newcatdir = opt.dmodel .. opt.model .. '/categories.txt'
if paths.filep(newcatdir) then
   print('Loading categories file from: ' .. newcatdir)
   network.classes, network.targets = readCatCSV(newcatdir)
end

if #network.classes == 0 then
   error('Categories file contains no categories')
end

print('Network has this list of categories, targets:')
for i=1,#network.classes do
   if opt.allcat then network.targets[i] = true end
   print(i..'\t'..network.classes[i]..'\t'..tostring(network.targets[i]))
end

classes = network.classes
--local testout = network.model:forward(torch.Tensor(1,3,256,256))
--print(testout[1]:size(1))

colorMap:init(opt, classes)
local colors = colorMap.getColors()

-- generating the <colourmap> out of the <colors> table
local colormap = imgraph.colormap(colors)
-- Initialize class Frame which can be used to read videos/camera
local frame
if string.sub(opt.input, 1, 3)  == 'cam' and tonumber(string.sub(opt.input,4,-1)) ~= nil then
   frame = assert(require('frame.framecamera'))
elseif opt.input:lower():match('%.jpe?g$') or opt.input:lower():match('%.png$') then
   frame = assert(require('frame.frameimage'))
elseif paths.dirp(opt.input) then
   frame = assert(require('frame.frameimage'))
else
   frame = assert(require('frame.framevideo'))
end

local source = {}
-- switch input sources
source.res = {
   HVGA  = {w =  320, h =  240},
   QHD   = {w =  640, h =  360},
   VGA   = {w =  640, h =  480},
   FWVGA = {w =  854, h =  480},
   HD    = {w = 1280, h =  720},
   FHD   = {w = 1920, h = 1080},
}
source.w = source.res[opt.camRes].w
source.h = source.res[opt.camRes].h
source.fps = opt.fps

-- opt.input is mandatory
-- source height and width gets updated by __init based on the input video
frame:init(opt, source)

-- Create a window for displaying output frames
win = qtwidget.newwindow
   ( source.w * opt.zoom + 75
   , source.h * opt.zoom
   , 'e-Lab Scene ParserParser'
 )
 
 --lbl = torch.Tensor(3,#classes * 100,200)
 

local qtimer = qt.QTimer()

-- Set font size to a visible dimension
win:setfontsize(12)

-- Show legends in the output window:
local dy = 20
if opt.limitClass then
   dy = (opt.zoom * source.h)/(#classSmall + 1)
   for i = 1,#classSmall do
      local y = (i-1)*dy
      win:rectangle(source.w * opt.zoom, y, 75, dy)
      win:setcolor(colorsSmall[i][1],colorsSmall[i][2],colorsSmall[i][3])
      win:fill()
      win:setcolor('black')
      win:moveto(source.w * opt.zoom + 5, y+dy/2)
      win:show(classSmall[i])
      --[[
      lbl[{{1}, {(i-1)*100+1, i*100}, {}}] = colorSmall[i][1]
      lbl[{{2}, {(i-1)*100+1, i*100}, {}}] = colorSmall[i][2]
      lbl[{{3}, {(i-1)*100+1, i*100}, {}}] = colorSmall[i][3]
      --]]
   end
else
   dy = (opt.zoom * source.h)/(#classes + 1)
   for i = 1,#classes do
      local y = (i-1)*dy
      win:rectangle(source.w * opt.zoom, y, 75, dy)
      win:setcolor(colors[i][1],colors[i][2],colors[i][3])
      win:fill()
      win:setcolor('black')
      win:moveto(source.w * opt.zoom + 5, y+dy/2)
      win:show(classes[i])
      --[[     
      lbl[{{1}, {(i-1)*100+1, i*100}, {}}] = colors[i][1]
      lbl[{{2}, {(i-1)*100+1, i*100}, {}}] = colors[i][2]
      lbl[{{3}, {(i-1)*100+1, i*100}, {}}] = colors[i][3]
      --]]

   
   end
end
   
--[[   
image.display(lbl)   
if (opt.out ~= nil and opt.out:len() > 0) then 
  image.savePNG(opt.out .. "/colors.png",lbl);
end
--]]

bPause = false
bOnce = false
bPause = false

-- profiling timers
local timer = torch.Timer()      -- whole loop
local totalTime
local tg = torch.Timer()         -- grabbing a frame
local grabTime
local tp = torch.Timer()         -- processing
local processTime
local tw = torch.Timer()
local winnerTime
local tc = torch.Timer()
local colormapTime
local td = torch.Timer()         -- displaying
local displayTime
local img
local fn
local outName
local inpName
local colored

local main = function()
   if win:valid() and (not bPause or bOnce) then
	if (bOnce) then
		bOnce = false
	end

      -- Reset timer to mark starting point to calculate fps
      timer:reset()

      -- Getting next frame
      tg:reset()
      img, fn, basename = frame.forward(img)
      if (fn == nil) then
        print('fn == nil')
        --  return false;
      end
      
      
      
      
      if opt.saveinp and opt.saveinp:len() > 0 then
        local inpDir = opt.saveinp .. '/' .. basename
        if (paths.dirp(inpDir) == false) then
          paths.mkdir(inpDir)
        end
        
        inpName = inpDir .. '/' .. fn;
      end
        
      if opt.out and opt.out:len() > 0 then
        local outDir = opt.out .. '/' .. basename
        if (paths.dirp(outDir) == false) then
          paths.mkdir(outDir)
        end
        
        outName = outDir .. '/' .. fn;
      end
      print (fn)

      win.name = ''
      grabTime = tg:time().real

      -- Processing the frame and forwarding it to network
      tp:reset()
      
       -- normalize the input:
       -- for i=1,img:size(1) do
       --    for c = 1,3 do
       --       img[i][c]:add(-network.stat.mean[c])
       --       img[i][c]:div( network.stat.std [c])
       --    end
       -- end
       
       local newW = source.w
       local newH = source.h
      -- 20160809 - optionally rescale to HD
      local imgHD
      if (opt.width > 0) then
        newW = opt.width
        newH = source.h * opt.width / source.w
        imgHD = image.scale( img[1], newW, newH, 'bilinear')
      else
        imgHD = image[1]
      end

      if imgHD:dim() == 3 then
         imgHD = imgHD:view(1, imgHD:size(1), newH, newW)
      end
      
      
      
      local scaledImg = torch.Tensor(1, 3, opt.ratio * newH, opt.ratio * newW)

      if opt.ratio == 1 then
         scaledImg[1] = imgHD[1]
      else
        print (imgHD:size(), scaledImg:size())
         scaledImg[1] = image.scale(imgHD[1],
                                    opt.ratio * newW,
                                    opt.ratio * newH,
                                    'bilinear')
      end

      if opt.dev == 'cuda' then
         scaledImgGPU = scaleImgGPU or torch.CudaTensor(scaledImg:size())
         scaledImgGPU:copy(scaledImg)
         scaledImg = scaledImgGPU
      end

      -- compute network on frame:
      distributions = network.model:forward(scaledImg):squeeze()

      processTime = tp:time().real

      --distributions = distributions

      -- Assigning classes to each pixels
      tw:reset()

      _, winners = distributions:max(1)

      if opt.dev == 'cuda' then
         cutorch.synchronize()
         winner = winners:squeeze():float()
      else
         winner = winners:squeeze()
      end

     
      -- Confirming whether rescaling is even necessary or not
      --local winner0 = winner:clone()
      if opt.ratio * source.h ~= winner:size(1) or
         opt.ratio * source.w ~= winner:size(2) then
         winner = image.scale(winner:float(),
                              source.w, -- * opt.ratio,
                              source.h, -- * opt.ratio,
                              'simple')
      end
      winnerTime = tw:time().real

      -- display output
      win:gbegin()

      tc:reset()
      -- colorize classes
      colored, colormap = imgraph.colorize(winner, colormap)
    
      if outName ~= nil and opt.save then
        image.savePNG(outName, colored)
        --torch.save(outName .. '.t7a',winner0, 'ascii')
      end

    local sum = colored:clone();
      -- add input image:
      sum:add(img[1]:float())

      colormapTime = tc:time().real

      td:reset()
      -- display frame:
      -- gui is turned off for faster display
      image.display{image=sum, win=win,
                    zoom=opt.zoom, gui=false,
                    min=0, max=colored:max()
                   }

      win:rectangle(source.w * opt.zoom, (source.h * opt.zoom)-dy, 75, dy)
      win:setcolor('white')
      win:fill()
      win:setcolor('black')
      win:moveto(source.w * opt.zoom + 5, (source.h * opt.zoom)-dy + 15)

      displayTime = td:time().real

      totalTime = timer:time().real

      local fps = string.format('%.2f', (1/totalTime)) .. ' fps'

      win:show(fps)
      -- display profiling on screen  
      win:setfont(qt.QFont{serif=false, italic=false, size=12})
      win:gend()

      if opt.verbose then
         print('Read    : ' .. string.format('%.0f', (grabTime*1000)),
               'Process : ' .. string.format('%.0f', (processTime*1000)),
               'Winner  : ' .. string.format('%.0f', (winnerTime*1000)),
               'Colormap: ' .. string.format('%.0f', (colormapTime*1000)),
               'Display : ' .. string.format('%.0f', (displayTime*1000)),
               '[fps]   : ' .. string.format('%.2f', (1/totalTime)))
      end

      collectgarbage()
     -- qtimer:start()
   end
end

----------------------------------------
-- Pausing or exiting demo
print("Press Spacebar to pause or")
print("      Right Arrow to skip forward or")
print("      Esc to exit")
qtimer.interval = 10
qtimer.singleShot = false
qt.connect(qtimer,'timeout()', main)

local prevState = true
qt.connect(win.listener,
         'sigKeyPress(QString, QByteArray, QByteArray)',
         function(_, keyValue)
            if keyValue == 'Key_E' then
              bPause = true;
              bOnce = true;
            elseif keyValue == 'Key_S' then
              if outName ~= nil  then
                image.savePNG(outName, colored)
              end
              if inpName ~= nil  then
                image.savePNG(inpName, img[1])
              end
            elseif keyValue == 'Key_Space' then
               print("Video paused; press enter to continue...")
               bPause = not bPause --io.read()
            elseif keyValue == 'Key_Escape' then
               os.exit()
            elseif keyValue == 'Key_Right' then
               for i = 1, 30 do
                  _ = frame:forward()
               end
               --qtimer:start()
            end
            qtimer:start()
         end
)
qtimer:start()
