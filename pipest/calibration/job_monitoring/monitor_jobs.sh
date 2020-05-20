#!/bin/bash

echo "I am monitoring my jobs"
touch job_monitor.txt
: > job_monitor.txt
arrjob_id=$1
q_stat=$(qstat)
while [ "$q_stat" != '' ]
do
   date +"%T" > time_now.txt
   mv job_monitor.txt m_old.txt
   qstat > m_new.txt
   qstat -t $arrjob_id > m_arr.txt
   cat m_old.txt time_now.txt m_new.txt m_arr.txt > job_monitor.txt
   rm m_old.txt
   rm m_new.txt
   rm time_now.txt
   rm m_arr.txt
   q_stat=$(qstat)
   if [ "$q_stat" = '' ]; then
       break
   fi
   sleep 300
done
echo "Monitoring concluded" 
