import { Box, Button, IconButton, Typography, useTheme } from "@mui/material";
import { tokens } from "../../theme";
import { mockTransactions } from "../../data/mockData";
import DownloadOutlinedIcon from "@mui/icons-material/DownloadOutlined";
import EmailIcon from "@mui/icons-material/Email";
import PointOfSaleIcon from "@mui/icons-material/PointOfSale";
import PersonAddIcon from "@mui/icons-material/PersonAdd";
import TrafficIcon from "@mui/icons-material/Traffic";
import CityBox from "../../components/CityBox";
import ThermostatIcon from '@mui/icons-material/Thermostat';
import SolarPowerRoundedIcon from '@mui/icons-material/SolarPowerRounded';
import BoltRoundedIcon from '@mui/icons-material/BoltRounded';
import CloudQueueRoundedIcon from '@mui/icons-material/CloudQueueRounded';
import FmdGoodIcon from '@mui/icons-material/FmdGood';
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



const Landing = (props) => {
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
      case "Power Output":
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
         return "km/h";
      case "Wind Direction":
         return "º";
      case "Irradiation":
         return "W/m2"
      case "Power Output":
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

  

  

  const parameters = ["Location Info","Irradiation","Power Output","Maximum Temperature","Minimum Temperature", "Feels Like","Humidity","Cloud Cover","Pressure","Wind Speed", "Wind Direction"]
  const chart_types = ["Bar Charts", "Line Charts","Pie Charts","Tables"]
  const cities = ["mumbai","delhi","bengaluru","pune","nagpur","hyderabad","kanpur","jaipur"]
  const chartCards = cities.length===0?'': cities.map(item=>{
   //let increase = parseFloat(data[item]["expected"])===0?0:((parseFloat(data[item]["current"])-parseFloat(data[item]["expected"]))/parseFloat(data[item]["expected"]))*100
   return(
      <Box
      gridColumn={"span 6"}
      backgroundColor={colors.primary[400]}
      display="flex"
      alignItems="center"
      justifyContent="center"
    >
      <CityBox
        title= {item.toUpperCase()}
        to={"/"+item}
        icon={
         ChartIcon(item)
        }
      />
    </Box>
   )

 })
  
  

  return (
    <Box m="20px">
      {/* HEADER */}
      <Box display="flex" justifyContent="space-between" alignItems="center">
        <Header title="DASHBOARD" subtitle="Welcome to your dashboard" />

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

export default Landing;
