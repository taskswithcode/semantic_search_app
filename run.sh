count=0
while :
do
    streamlit run app.py --server.port 8502  --server.enableCORS false --server.enableXsrfProtection false
    ((count=count+1))
    echo "Restarted $count times"
done

