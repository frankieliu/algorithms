# Steps

## get links
1. start with a search at jobs.apple.com
   1. filter on location and key ai
   [link](https://jobs.apple.com/en-us/search?location=cupertino-CUP+sunnyvale-SVL+santa-clara-SNC&page=3&key=ai)

1. for each page, use link grabber and save to notepad

## filter links
1. `python 01_filter.py`
filter.py reads off a links files (above)
   - modify the file to point to the date
   - outputs to a pkl file containing df (job number / title / team)
   - `<date>_01_job_number.pkl` 

## dedup jobs from previous
1. `python 02_dedup.py`
   - modify the file to point to old and new date
   - outputs only new jobs to a pkl file
   - outputs combined new + old jobs to a pkl file (to be used on next dedup)
   - `<date>_02_job_number_dedup.pkl`
   - `<date>_02_job_number_cum.pkl`

## scrape the data from the website
1. `mkdir <date>_03_scrape`
1. `python 03_scrape_site.py` 
   - `<date>_03_scrape/`

## extract the sections and save to database
1. `python 04_extract_sections.py`
   - `<date>_04_job_descriptions.pkl`

## save to a database
1. `python 05_to_db.py`
   - save to apple.db
   - on new table named `jobs_<date>`

## examine results
1. open db browser (sqlite)
1. C-o (open database) ../data/apple.db
1. `cd ./sql; cp alter_.. alter_..`
1. Goto Execute SQL/ Load the Sql, and execute it
1. modify alter to add status (requirements), score (desirability), comment

## get parsed information from email sent
1. `python processed.py`
1. join on the table
