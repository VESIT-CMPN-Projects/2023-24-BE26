import { Box, Button, IconButton, Typography, useTheme } from "@mui/material";
import { tokens } from "../../theme";
import { mockTransactions } from "../../data/mockData";
import DownloadOutlinedIcon from "@mui/icons-material/DownloadOutlined";
import EmailIcon from "@mui/icons-material/Email";
import PointOfSaleIcon from "@mui/icons-material/PointOfSale";
import PersonAddIcon from "@mui/icons-material/PersonAdd";
import TrafficIcon from "@mui/icons-material/Traffic";
import ThermostatIcon from '@mui/icons-material/Thermostat';
import SolarPowerRoundedIcon from '@mui/icons-material/SolarPowerRounded';
import BoltRoundedIcon from '@mui/icons-material/BoltRounded';
import CloudQueueRoundedIcon from '@mui/icons-material/CloudQueueRounded';
import FmdGoodIcon from '@mui/icons-material/FmdGood';
import QueryBuilderIcon from '@mui/icons-material/QueryBuilder';
import AirIcon from '@mui/icons-material/Air';
import WaterDropIcon from '@mui/icons-material/WaterDrop';
import Header from "../../components/Header";
import LineChart from "../../components/LineChart";
import GeographyChart from "../../components/GeographyChart";
import BarChart from "../../components/BarChart";
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import StatBox from "../../components/StatBox";
import ChartBox from "../../components/ChartBox";
import ProgressCircle from "../../components/ProgressCircle";
import InsertChartIcon from '@mui/icons-material/InsertChart';
import PieChartIcon from '@mui/icons-material/PieChart';
import InsightsIcon from '@mui/icons-material/Insights';
import TableChartIcon from '@mui/icons-material/TableChart';

import { useEffect, useState } from "react";



const Dashboard = (props) => {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);
  if(props.city === ""){
   props.city = "mumbai"
  }

  const Icon = (param)=>{

   switch(param){
      case "Minimum Temperature":
      case "Maximum Temperature":
      case "Pressure":
      case "Feels Like":
         return( <ThermostatIcon 
         sx={{ color: colors.greenAccent[600], fontSize: "26px" }}
         />)
      case "Wind Speed":
      case "Wind Direction":
         return( <AirIcon
            sx={{ color: colors.greenAccent[600], fontSize: "26px" }}
            />)
      case "Humidity":
         return( <WaterDropIcon
            sx={{ color: colors.greenAccent[600], fontSize: "26px" }}
            />)
      case "Irradiation":
         return( <SolarPowerRoundedIcon
            sx={{ color: colors.greenAccent[600], fontSize: "26px" }}
            />)
      case "Power":
         return( <BoltRoundedIcon
            sx={{ color: colors.greenAccent[600], fontSize: "26px" }}
            />)
      case "Cloud Cover":
         return( <CloudQueueRoundedIcon
            sx={{ color: colors.greenAccent[600], fontSize: "26px" }}
            />)
      default:
         return (<></>)
   }
}

const ChartIcon = (param)=>{
   switch(param){
      case "Bar Charts":
         return (
            <InsertChartIcon 
            sx={{ color: colors.greenAccent[600], fontSize: "36px" }}
            />
         )
      case "Line Charts":
         return (<InsightsIcon 
            sx={{ color: colors.greenAccent[600], fontSize: "36px" }}
         />)
      case "Pie Charts":
         return (
            <PieChartIcon
            sx={{ color: colors.greenAccent[600], fontSize: "36px" }}
            />
         )
      case "Tables":
         return (
            <TableChartIcon
            sx={{ color: colors.greenAccent[600], fontSize: "36px" }}
            />
         )
      default:
         return (<></>)

   }

}

const Unit = (param)=>{
   switch(param){
      case "Minimum Temperature":
      case "Maximum Temperature":
      case "Feels Like":
         return "ºC";
        
      case "Pressure":
         return "mbar";
      case "Wind Speed":
         return "m/s";
      case "Wind Direction":
         return "º";
      case "Irradiation":
         return "W/m2"
      case "Power":
         return "kWh"
      case "Humidity":
      case "Cloud Cover":
         return "%";

      default:
         return "";
   }
}

const trending =(increase)=>{
   if(increase >=0){
     return (<TrendingUpIcon sx= {{ marginLeft:"4px"  , color: "#6870fa"}}
     />);
   }else{
     return (<TrendingDownIcon sx= {{ marginLeft:"4px"  , color: "#f25c6e"}}
     />);
   }
 }

  let [data, setData] = useState([])
  useEffect(()=>{

   getData();
  },[])

  let getData = async () =>{
      let response = await fetch('http://127.0.0.1:8000/api/city/'+props.city)
      let res = await response.json()
      //console.log(res[0]["Maximum Temperature"])
      console.log(res[0])
      setData(res[0])
      
  }

  const parameters = ["Maximum Temperature","Minimum Temperature", "Feels Like","Humidity","Cloud Cover","Pressure","Wind Speed", "Wind Direction"]
  const chart_types = [["Bar Charts","bar"], ["Pie Charts","pie"]]//,["Line Charts","line"],["Tables","table"]]
  
  const chartCards = data.length===0?'': chart_types.map(item=>{
   //let increase = parseFloat(data[item]["expected"])===0?0:((parseFloat(data[item]["current"])-parseFloat(data[item]["expected"]))/parseFloat(data[item]["expected"]))*100
   return(
      <Box
      gridColumn={"span 6"}
      backgroundColor={colors.primary[400]}
      display="flex"
      alignItems="center"
      justifyContent="center"
    >
      <ChartBox
        title= {item[0]}
        to={"/"+item[1]+"/"+(item[1]==='pie'?'':props.city)}
        icon={
         ChartIcon(item[0])
        }
      />
    </Box>
   )

 })

 const locationcard = data.length===0?'':()=>{
  console.log("Yess")
  return(
    <Box
    gridColumn={"span 12"}
    backgroundColor={colors.primary[400]}
    display="flex"
    alignItems="center"
    justifyContent="center"
  >
    <StatBox
      title= {data["Location Info"]["name"]}
      subtitle={data["Location Info"]["display_name"]}
      progress={parseFloat(data["Location Info"]["lat"]).toFixed(2)+"º ,"+parseFloat(data["Location Info"]["lon"]).toFixed(2)+"º "}
      trending = {''}
      increase={''}
      icon={
       <FmdGoodIcon 
       sx={{ color: colors.greenAccent[600], fontSize: "26px" }}
       />
      }
    />
  </Box>
  )
 }
  const cards = data.length===0?'': parameters.map(item=>{
   let increase = parseFloat(data[item]["expected"])===0?0:((parseFloat(data[item]["current"])-parseFloat(data[item]["expected"]))/parseFloat(data[item]["expected"]))*100
   
   return(
      <Box
      gridColumn={item==="Location Info"?"span 12":(item==="Irradiation"||item==="Power Output"?"span 6": "span 3")}
      backgroundColor={colors.primary[400]}
      display="flex"
      alignItems="center"
      justifyContent="center"
    >
      <StatBox
        title= {item==="Location Info"?data[item]["name"]:data[item]["current"]}
        subtitle={item==="Location Info"?data[item]["display_name"]:item}
        progress={item==="Location Info"?parseFloat(data[item]["lat"]).toFixed(2)+"º ,"+parseFloat(data[item]["lon"]).toFixed(2)+"º ":Unit(item)}
        trending = {item==="Location Info"?"":trending(increase)}
        increase={item==="Location Info"?"":increase>=0?"+"+increase.toFixed(2):increase.toFixed(2)}
        icon={
         item==="Location Info"?<FmdGoodIcon 
         sx={{ color: colors.greenAccent[600], fontSize: "26px" }}
         />:Icon(item)
        }
      />
    </Box>
   )

 })

 const solarcards = data.length===0?'': data["Solar Power"].map(item=>{
  let increase = parseFloat(item.expected===0?0:((parseFloat(item.current)-parseFloat(item.expected))/parseFloat(item.expected))*100)
  
  return(
     <Box
     gridColumn={"span 6"}
     backgroundColor={colors.primary[400]}
     display="flex"
     alignItems="center"
     justifyContent="center"
   >
     <StatBox
       title= {item.current}
       subtitle={item.Name}
       progress={Unit(item.Icon)}
       trending = {trending(increase)}
       increase={increase.toFixed(2)}
       icon={
        Icon(item.Icon)
       }
     />
   </Box>
  )

})

const windcards = data.length===0?'': data["Wind Power"].map(item=>{
  
  let increase = parseFloat(item.expected===0?0:((parseFloat(item.current)-parseFloat(item.expected))/parseFloat(item.expected))*100)
  
  return(
     <Box
     gridColumn={"span 6"}
     backgroundColor={colors.primary[400]}
     display="flex"
     alignItems="center"
     justifyContent="center"
   >
     <StatBox
       title= {item.current}
       subtitle={item.Name}
       progress={Unit(item.Icon)}
       trending = {trending(increase)}
       increase={increase.toFixed(2)}
       icon={
        Icon(item.Icon)
       }
     />
   </Box>
  )

})

 const timecards = data.length===0?'': data["Time"].map(item=>{
  //let increase = parseFloat(data[item]["expected"])===0?0:((parseFloat(data[item]["current"])-parseFloat(data[item]["expected"]))/parseFloat(data[item]["expected"]))*100
  
  return(
     <Box
     gridColumn={"span 4"}
     backgroundColor={colors.primary[400]}
     display="flex"
     alignItems="center"
     justifyContent="center"
   >
     <StatBox
       title= {item.Time}
       subtitle={item.Date}
       progress={item.Name}
       trending = {""}
       increase={""}
       icon={
        <QueryBuilderIcon 
        sx={{ color: colors.greenAccent[600], fontSize: "26px" }}
        />
       }
     />
   </Box>
  )

})
  
console.log(locationcard)
  return (
    <Box m="20px" >
      {/* HEADER */}
      <Box display="flex" justifyContent="space-between" alignItems="center">
        <Header title="DASHBOARD" subtitle="As of time : " sx={{width:"100%"}}/>
        <Header title=" " subtitle="*LST = Local Solar Time" />
        <Box display={"none"}>
          <Button
            sx={{
              backgroundColor: colors.blueAccent[700],
              color: colors.grey[100],
              fontSize: "14px",
              fontWeight: "bold",
              padding: "10px 20px",
            }}
          >
            <DownloadOutlinedIcon sx={{ mr: "10px", display:"none" }} />
            Download Reports
          </Button>
        </Box>
      </Box>

      {/* GRID & CHARTS */}
      <Box
        display="grid"
        gridTemplateColumns="repeat(12, 1fr)"
        gridAutoRows="140px"
        gap="20px"
      >
        {/* ROW 1 */}
        {timecards}
        {data.length===0?'':locationcard()}
        {solarcards}
        {windcards}
        {cards}
       {/* <Box
          gridColumn="span 3"
          backgroundColor={colors.primary[400]}
          display="flex"
          alignItems="center"
          justifyContent="center"
        >
          <StatBox
            title="12,361"
            subtitle="Emails Sent"
            progress="0.75"
            increase="+14%"
            icon={
              <EmailIcon
                sx={{ color: colors.greenAccent[600], fontSize: "26px" }}
              />
            }
          />
        </Box>
        
        <Box
          gridColumn="span 3"
          backgroundColor={colors.primary[400]}
          display="flex"
          alignItems="center"
          justifyContent="center"
        >
          <StatBox
            title="431,225"
            subtitle="Sales Obtained"
            progress="0.50"
            increase="+21%"
            icon={
              <PointOfSaleIcon
                sx={{ color: colors.greenAccent[600], fontSize: "26px" }}
              />
            }
          />
        </Box>
        <Box
          gridColumn="span 3"
          backgroundColor={colors.primary[400]}
          display="flex"
          alignItems="center"
          justifyContent="center"
        >
          <StatBox
            title="32,441"
            subtitle="New Clients"
            progress="0.30"
            increase="+5%"
            icon={
              <PersonAddIcon
                sx={{ color: colors.greenAccent[600], fontSize: "26px" }}
              />
            }
          />
        </Box>
        <Box
          gridColumn="span 3"
          backgroundColor={colors.primary[400]}
          display="flex"
          alignItems="center"
          justifyContent="center"
        >
          <StatBox
            title="1,325,134"
            subtitle="Traffic Received"
            progress="0.80"
            increase="+43%"
            icon={
              <TrafficIcon
                sx={{ color: colors.greenAccent[600], fontSize: "26px" }}
              />
            }
          />
        </Box> */}

        {/* ROW 2 */}
        {/*<Box
          gridColumn="span 8"
          gridRow="span 2"
          backgroundColor={colors.primary[400]}
        >
          <Box
            mt="25px"
            p="0 30px"
            display="flex "
            justifyContent="space-between"
            alignItems="center"
          >
            <Box>
              <Typography
                variant="h5"
                fontWeight="600"
                color={colors.grey[100]}
              >
                Revenue Generated
              </Typography>
              <Typography
                variant="h3"
                fontWeight="bold"
                color={colors.greenAccent[500]}
              >
                $59,342.32
              </Typography>
            </Box>
            <Box>
              <IconButton>
                <DownloadOutlinedIcon
                  sx={{ fontSize: "26px", color: colors.greenAccent[500] }}
                />
              </IconButton>
            </Box>
          </Box>
          <Box height="250px" m="-20px 0 0 0">
            <LineChart isDashboard={true} />
          </Box>
        </Box>
        <Box
          gridColumn="span 4"
          gridRow="span 2"
          backgroundColor={colors.primary[400]}
          overflow="auto"
        >
          <Box
            display="flex"
            justifyContent="space-between"
            alignItems="center"
            borderBottom={`4px solid ${colors.primary[500]}`}
            colors={colors.grey[100]}
            p="15px"
          >
            <Typography color={colors.grey[100]} variant="h5" fontWeight="600">
              Recent Transactions
            </Typography>
          </Box>
          {mockTransactions.map((transaction, i) => (
            <Box
              key={`${transaction.txId}-${i}`}
              display="flex"
              justifyContent="space-between"
              alignItems="center"
              borderBottom={`4px solid ${colors.primary[500]}`}
              p="15px"
            >
              <Box>
                <Typography
                  color={colors.greenAccent[500]}
                  variant="h5"
                  fontWeight="600"
                >
                  {transaction.txId}
                </Typography>
                <Typography color={colors.grey[100]}>
                  {transaction.user}
                </Typography>
              </Box>
              <Box color={colors.grey[100]}>{transaction.date}</Box>
              <Box
                backgroundColor={colors.greenAccent[500]}
                p="5px 10px"
                borderRadius="4px"
              >
                ${transaction.cost}
              </Box>
            </Box>
          ))}
        </Box>
*/}
        {/* ROW 3 */}
        {/*<Box
         gridColumn="span 4"
          gridRow="span 2"
          backgroundColor={colors.primary[400]}
          p="30px"
        >
          <Typography variant="h5" fontWeight="600">
            Campaign
          </Typography>
          <Box
            display="flex"
            flexDirection="column"
            alignItems="center"
            mt="25px"
          >
            <ProgressCircle size="125" />
            <Typography
              variant="h5"
              color={colors.greenAccent[500]}
              sx={{ mt: "15px" }}
            >
              $48,352 revenue generated
            </Typography>
            <Typography>Includes extra misc expenditures and costs</Typography>
          </Box>
        </Box>
        <Box
          gridColumn="span 4"
          gridRow="span 2"
          backgroundColor={colors.primary[400]}
        >
          <Typography
            variant="h5"
            fontWeight="600"
            sx={{ padding: "30px 30px 0 30px" }}
          >
            Sales Quantity
          </Typography>
          <Box height="250px" mt="-20px">
            <BarChart isDashboard={true} />
          </Box>
        </Box>
        <Box
          gridColumn="span 4"
          gridRow="span 2"
          backgroundColor={colors.primary[400]}
          padding="30px"
        >
          <Typography
            variant="h5"
            fontWeight="600"
            sx={{ marginBottom: "15px" }}
          >
            Geography Based Traffic
          </Typography>
          <Box height="200px">
            <GeographyChart isDashboard={true} />
          </Box>
        </Box>*/}
      </Box>

      <Box display="flex" justifyContent="space-between" marginTop={"5%"} alignItems="center">
        <Header title="Data Visualization" subtitle="Uncover Trends in Energy Production" />

        <Box display={"none"}>
          <Button
            sx={{
             
              backgroundColor: colors.blueAccent[700],
              color: colors.grey[100],
              fontSize: "14px",
              fontWeight: "bold",
              padding: "10px 20px",
            }}
          >
            <DownloadOutlinedIcon sx={{ mr: "10px", display:"none" }} />
            Download Reports
          </Button>
        </Box>
      </Box>

      <Box
      marginBottom={"5%"}
        display="grid"
        gridTemplateColumns="repeat(12, 1fr)"
        gridAutoRows="140px"
        gap="20px"
        
      >

         {chartCards}
      </Box>

    </Box>
  );
};

export default Dashboard;
