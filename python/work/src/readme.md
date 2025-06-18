# Steps

## get links
1. start with a serach at jobs.apple.com
   1. filter on location and key ai
   [link](https://jobs.apple.com/en-us/search?location=cupertino-CUP+sunnyvale-SVL+santa-clara-SNC&page=3&key=ai)

1. for each page, use link grabber and save to notepad

## filter links
1. filter.py reads off a links files (above)
   - outputs to a pkl file containing df (job number / title / team)

## dedup jobs from previous
1. python dedup.py
   - outputs only new jobs to a pkl file
   - outputs compbied new + old jobs to a pkl file (to be used on next dedup)

## scrape the data from the website
1. create a new directory `<date>_scrape`
1. `python scrape_site.py` 

## extract the sections and save to database
1. `python extract_sections.py`
   - outputs to a pkl file named `<date>_job_descriptions.pkl`

## save to a database
1. `python to_db.py`
   - save to apple.db
   - on new table named `jobs_<date>`
