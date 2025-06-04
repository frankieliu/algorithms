create table jobs_2025_06_02_265000 as 
select * from `jobs.2025.06.02`
where pay_max > 265000
order by team, pay_max desc, title;