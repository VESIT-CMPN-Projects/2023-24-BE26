import { Box, Typography, useTheme } from "@mui/material";
import { tokens } from "../theme";
import ProgressCircle from "./ProgressCircle";
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import { Link } from "react-router-dom";
const CityBox = ({ title,subtitle,to, icon }) => {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);
  
  return (
    <Box width="100%" m="0 30px" >
      <Box display="flex" justifyContent="space-between">
        <Box>
          
          
          <Typography
            variant="h4"
            fontWeight="bold"
            sx={{ color: colors.grey[100] }}
          >
            {title} 
          </Typography>
        </Box>
        <Box >
        <Typography
            variant="h4"
            fontWeight="bold"
            sx={{ color: colors.greenAccent[500] }}
          >
            {icon}
          </Typography>
         
        </Box>
        
      </Box>
      
      <Box display="flex" justifyContent="space-between" mt="2px">
     
        <Typography variant="h6" sx={{ color: colors.greenAccent[500] }}>
          {subtitle}
  </Typography>
        <Box display="flex" justifyContent="space-between" >
       
        <Link to={to}
        
        sx={{ color: colors.greenAccent[400] }}>
 <Typography
          variant="h5"
          fontStyle="italic"
          sx={{ color: colors.greenAccent[600] }}
        >
          View More
        </Typography>

        </Link>
        
        
         
        
        
         </Box>
  </Box>
    </Box>
  );
};

export default CityBox;
