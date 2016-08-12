require 'diz'
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
cd ('../ENet-training/train')

md=torch.load('model-b314e-3N373.n7a','ascii')
data=torch.load('data-050v.t7a','ascii')
lb=md:forward(data.testData.data)
cd '../visualize'

colorMap=require('colorMap')
catfile = '../train/categories.txt'

network={}
network.classes, network.targets = readCatCSV(catfile)

opt = torch.load('opt.t7a','ascii')

classes = network.classes
opt.dataset = 'css'  -- my dataset!!!!
colorMap:init(opt, classes)

require 'imgraph'

colors=colorMap.getColors()
colormap=imgraph.colormap(colors)

_, winners = lb:max(2)

wn6 = winners[6]:squeeze()
wn6f=image.scale(wn6:float(), 640, 360, 'simple')

colored, cp = imgraph.colorize(wn6f, colormap:float())
imshow(colored+d6)

