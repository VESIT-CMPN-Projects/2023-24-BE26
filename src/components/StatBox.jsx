import { Box, Typography, useTheme } from "@mui/material";
import { tokens } from "../theme";
import ProgressCircle from "./ProgressCircle";
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';

const StatBox = ({ title, subtitle, icon, progress, increase, trending }) => {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);
  
  return (
    <Box width="100%" m="0 30px">
      <Box display="flex" justifyContent="space-between">
        <Box>
          {icon}
          
          <Typography
            variant="h3"
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
            {progress}
          </Typography>
         
        </Box>
        
      </Box>
      
      <Box display="flex" justifyContent="space-between" mt="2px">
     
        <Typography variant="h6" sx={{ color: colors.greenAccent[500] }}>
          {subtitle}
        </Typography>
        <Box display="flex" justifyContent="space-between" >
        <Typography
          variant="h5"
          fontStyle="italic"
          sx={{ color: colors.greenAccent[600] }}
        >
          {increase?increase+"%":""}
        </Typography>
        {trending}
        
         </Box>
      </Box>
    </Box>
  );
};

export default StatBox;
