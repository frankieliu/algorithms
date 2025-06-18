drop table jobs_250617_sorted;
create table jobs_250617_sorted as
select j1.*, j2.action 
from jobs_2025_06_02_26500_score_6 as j1
left join jobs_250617_action as j2
on j1.number = j2.number
-- where score > 5
order by score desc, status asc, j2.number desc, `title-short`
;