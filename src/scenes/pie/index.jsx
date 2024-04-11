import { Box, Button, IconButton, Typography, useTheme } from "@mui/material";
import Header from "../../components/Header";
import PieChart from "../../components/PieChart";
import { mockPieData as data } from "../../data/mockData";
import { useEffect, useState } from "react";
import { tokens } from "../../theme";

import QueryBuilderIcon from '@mui/icons-material/QueryBuilder';



import StatBox from "../../components/StatBox";

const Pie = () => {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);
  let [data, setData] = useState([])
  useEffect(()=>{
  
   getData();
  },[])

  let [solarData, setsolarData] = useState([])
  let [windData, setwindData] = useState([])
  let [fullsolarData, setfullsolarData] = useState([])
  let [fullwindData, setfullwindData] = useState([])
  let [locData, setlocData] = useState([])
  
  
  let getData = async () =>{
    
      let response = await fetch('http://127.0.0.1:8000/api/city_pie/')
      let res = await response.json()
      const SolarPieData = [];
      const WindPieData = [];
      const FulldaySolarPieData = [];
      const FulldayWinPieData = [];
      setlocData(res.Time)
      res.Main.map((item)=>{
        let sol = {
          id:item.city,
          label : item.city,
          value : item["Solar Power"],

        }
        let fullsol = {
          id:item.city,
          label : item.city,
          value : item["Solar Power Fullday"],

        }
        let win = {
          id:item.city,
          label : item.city,
          value : item["Wind Power"],

        }
        let fullwin = {
          id:item.city,
          label : item.city,
          value : item["Wind Power Fullday"],

        }
        SolarPieData.push(sol)
        FulldaySolarPieData.push(fullsol)
        WindPieData.push(win)
        FulldayWinPieData.push(fullwin)
      })

      setsolarData(SolarPieData)
      setfullsolarData(FulldaySolarPieData)
      setfullwindData(FulldayWinPieData)
      setwindData(WindPieData)
      
      
      
      setData(res.Main)
      
  }

  const timecards = data.length===0?'': locData.map(item=>{
    //let increase = parseFloat(data[item]["expected"])===0?0:((parseFloat(data[item]["current"])-parseFloat(data[item]["expected"]))/parseFloat(data[item]["expected"]))*100
    
    return(
       <Box
       gridColumn={"span 6"}
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
    <Box m="20px">
      <Header title="Current Solar Generation" subtitle="(30 panels per Day) (in kWh)" />
      <Box height="75vh">
        <PieChart inputData={solarData} />
      </Box>
    </Box>
    <Box m="20px">
      <Header title="Avg Exp Solar Generation for Today" subtitle="(30 panels per Day) (in kWh)" />
      <Box height="75vh">
        <PieChart inputData={fullsolarData} />
      </Box>
    </Box>
    <Box m="20px">
      <Header title="Current Wind Generation" subtitle="(1 turbine per Day) (in kWh)" />
      <Box height="75vh">
        <PieChart inputData={windData} />
      </Box>
    </Box>
    <Box m="20px">
      <Header title="Avg Exp Wind Generation for Today" subtitle="(1 turbine per Day) (in kWh)" />
      <Box height="75vh">
        <PieChart inputData={fullwindData} />
      </Box>
    </Box>
    </>
    
  );
};

export default Pie;
