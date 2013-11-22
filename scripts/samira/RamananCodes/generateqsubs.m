
fid=fopen('qsub.bat', 'w');
imagedir=dir(fullfile('CKiccv\*.png'));
for i=1:length(imagedir)%
    i
    linestr = ['qsub -v a=' num2str(i) ' -l nodes=1,walltime=20:00:00 matlab.bat'];
    %linestr = ['mv ' num2str(i) ' censusresults/' num2str(i)];
    linestr=[linestr '\n'];
    fprintf(fid,linestr);
end
fclose(fid);