function color2idx(c)
   if c == 40 then
      return 2
   elseif c == 30 then
      return 3
   elseif c == 60 then
      return 4
   elseif c == 20 then
      return 5
   elseif c == 70 then
      return 6
   elseif c == 26 then
      return 7
   elseif c == 10 then
      return 8
   elseif c == 14 then
      return 9
   else
      return 1
   end
end