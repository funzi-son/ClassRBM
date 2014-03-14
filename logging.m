function logging( file,log_dat)
% Log all values in data to the file
% sontran2013
stt  = exist(file,'file');
if stt == 0
    data = zeros(0,0);
else
    load(file);
end
data = [data;log_dat];
save(file,'data');
end

