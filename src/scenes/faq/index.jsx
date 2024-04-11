import { Box, useTheme } from "@mui/material";
import Header from "../../components/Header";
import Accordion from "@mui/material/Accordion";
import AccordionSummary from "@mui/material/AccordionSummary";
import AccordionDetails from "@mui/material/AccordionDetails";
import Typography from "@mui/material/Typography";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import { tokens } from "../../theme";
import { faqData } from "../../data/mockData";

const FAQ = () => {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);
  const faq = faqData.map((item)=>{
    return (<Accordion defaultExpanded>
      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
        <Typography color={colors.greenAccent[500]} variant="h5">
          {item.q}
        </Typography>
      </AccordionSummary>
      <AccordionDetails>
        <Typography>
        {item.a}
        </Typography>
      </AccordionDetails>
    </Accordion>);
  });
  return (
    <Box m="20px">
      <Header title="FAQ" subtitle="Frequently Asked Questions Page" />
    {faq}
      
    </Box>
  );
};

export default FAQ;
