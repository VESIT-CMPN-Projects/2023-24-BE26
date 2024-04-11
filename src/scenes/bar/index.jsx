
import { Box, Button, IconButton, Typography, useTheme } from "@mui/material";
import Header from "../../components/Header";
import BarChart from "../../components/BarChart";
import { actualBarData as datamock } from "../../data/mockData";
import { useEffect, useState } from "react";
import { tokens } from "../../theme";
import StatBox from "../../components/StatBox";
import QueryBuilderIcon from '@mui/icons-material/QueryBuilder';
import FmdGoodIcon from '@mui/icons-material/FmdGood';
const Bar = (props) => {

  let [data, setData] = useState([])
useEffect(()=>{

 getData();
},[])

let [timedata, setTimedata] = useState([])
let [locdata, setLocdata] = useState([])
if(props.city === ""){
  props.city = "mumbai"
 }
let getData = async () =>{
  
    let response = await fetch('http://127.0.0.1:8000/api/city_bar/'+props.city)
    let res = await response.json()
    console.log(res.Time)
 
    setData(res.Main)
    setTimedata(res.Time)
    setLocdata(res.Location)
    
}

const theme = useTheme();
  const colors = tokens(theme.palette.mode);
  console.log(locdata)
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
        title= {locdata.name}
        subtitle={locdata.display_name}
        progress={parseFloat(locdata.lat).toFixed(2)+"º ,"+parseFloat(locdata.lon).toFixed(2)+"º "}
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
  const timecards = timedata.length===0?'': timedata.map(item=>{
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
    
  return (
   <>
    <Box
        display="grid"
        gridTemplateColumns="repeat(12, 1fr)"
        gridAutoRows="140px"
        gap="20px"
      >{timecards}
     
      </Box>

      
    <Box m="20px" sx={{  backgroundColor: colors.primary[400] }}>
      <Header title="Power Consumption vs Generation" subtitle={"For "+props.city} />
      <Box height="100vh">
        <BarChart Inputdata={data} keys={['Wind','Solar']} index={'Source'} />
      </Box>
    </Box>
    </>
  );
};

export default Bar;
