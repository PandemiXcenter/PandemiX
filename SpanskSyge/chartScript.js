// Initialize flags
var showRatioK = false;

// --- Load data from csv ---
// Initialize arrays for all data
var allDatesRaw = []
var allDates = []
var cases0_1 = []
var cases1_4 = []
var cases5_14 = []
var cases15_64 = []
var cases65 = []
var casesSum = []
var ratioK0_1 = []
var ratioK1_4 = []
var ratioK5_14 = []
var ratioK15_64 = []
var ratioK65 = []
var ratioKSum = []

doneLoading = false;
var curMinDate = new Date('1915-01-01');
var curMaxDate = new Date('1920-01-01');

d3.csv("Spansk_Syge_I_Cph.csv").then(function(data) {
    for (let k = 0; k < data.length; k++) {
        const curRow = data[k];
        cases0_1.push(curRow.cases0_1);
        cases1_4.push(curRow.cases1_5);
        cases5_14.push(curRow.cases5_15);
        cases15_64.push(curRow.cases15_65);
        cases65.push(curRow.cases65);
        casesSum.push(curRow.casesSum);
        ratioK0_1.push(curRow.ratioK0_1);
        ratioK1_4.push(curRow.ratioK1_4);
        ratioK5_14.push(curRow.ratioK5_14);
        ratioK15_64.push(curRow.ratioK15_64);
        ratioK65.push(curRow.ratioK65);
        ratioKSum.push(curRow.ratioKAll);
        allDatesRaw.push(curRow.date);
    }
    
    doneLoading = true;
    interpretDate();

    updateChart();


    // Update the sliders, to ensure label is correct
    xlim1_slider.onchange();
    xlim2_slider.onchange();
    
});

// Load deaths
var allDatesDeathRaw = [];
var allDatesDeath = [];
var allDeath = [];
d3.csv("Spansk_Syge_Cph_Doedsfald.csv").then(function(data) {
    for (let k = 0; k < data.length; k++) {
        const curRow = data[k];
        allDatesDeathRaw.push(curRow.Date);
        allDeath.push(curRow.Deaths);
    }
    
    doneLoading = true;
    interpretDate();
    updateChart();


    // // Update the sliders, to ensure label is correct
    // xlim1_slider.onchange();
    // xlim2_slider.onchange();
    
});

// Load age-split deaths
var deathDatesRaw = [];
var deathDates = [];
var deaths0_4 = []
var deaths5_14 = []
var deaths15_64 = []
var deaths65 = []
var deathsSum = []
var ratioDeaths0_4 = []
var ratioDeaths5_14 = []
var ratioDeaths15_64 = []
var ratioDeaths65 = []
var ratioDeathsSum = []

// Hardcoded age-structured population for 1918 data from Hansen (1919)
var pop0_4 = 10014 + 36560
var pop5_14 = 89006
var pop15_64 = 360096
var pop65 = 30317
var popSum = pop0_4 + pop5_14 + pop15_64 + pop65

d3.csv("HansenSide7TilWeb.csv").then(function(data) {
    for (let k = 0; k < data.length; k++) {
        const curRow = data[k];
        let cur0 = parseInt(curRow.m0) + parseInt(curRow.k0)
        let cur5  = parseInt(curRow.m5) + parseInt(curRow.k5)
        let cur10 = parseInt(curRow.m10) + parseInt(curRow.k10)
        let cur15 = parseInt(curRow.m15) + parseInt(curRow.k15)
        let cur20 = parseInt(curRow.m20) + parseInt(curRow.k20)
        let cur25 = parseInt(curRow.m25) + parseInt(curRow.k25)
        let cur30 = parseInt(curRow.m30) + parseInt(curRow.k30)
        let cur35 = parseInt(curRow.m35) + parseInt(curRow.k35)
        let cur40 = parseInt(curRow.m40) + parseInt(curRow.k40)
        let cur45 = parseInt(curRow.m45) + parseInt(curRow.k45)
        let cur50 = parseInt(curRow.m50) + parseInt(curRow.k50)
        let cur55 = parseInt(curRow.m55) + parseInt(curRow.k55)
        let cur60 = parseInt(curRow.m60) + parseInt(curRow.k60)
        let cur65 = parseInt(curRow.m65) + parseInt(curRow.k65)
        let cur70 = parseInt(curRow.m70) + parseInt(curRow.k70)
        let cur75 = parseInt(curRow.m75) + parseInt(curRow.k75)
        let cur80 = parseInt(curRow.m80) + parseInt(curRow.k80)
        let cur85 = parseInt(curRow.m85) + parseInt(curRow.k85)
        let curU = parseInt(curRow.mu) + parseInt(curRow.ku)

        let cur0_4 = cur0;
        let cur5_14 = cur5+cur10;
        let cur15_64 = cur15+cur20+cur25+cur30+cur35+cur40+cur45+cur50+cur55+cur60;
        let cur65plus = cur65+cur70+cur75+cur80+cur85+curU;
        let curSum = cur0_4+cur5_14+cur15_64+cur65plus;
        deaths0_4.push(cur0_4);
        deaths5_14.push(cur5_14);
        deaths15_64.push(cur15_64);
        deaths65.push(cur65plus);
        deathsSum.push(curSum);
        
        let curRatio0_4 = 1000 * cur0_4 / pop0_4;
        let curRatio5_14 = 1000 * cur5_14 / pop5_14;
        let curRatio15_64 = 1000 * cur15_64 / pop15_64;
        let curRatio65 = 1000 * cur65plus / pop65;
        let curRatioSum = 1000 * curSum / popSum;
        ratioDeaths0_4.push(curRatio0_4);
        ratioDeaths5_14.push(curRatio5_14);
        ratioDeaths15_64.push(curRatio15_64);
        ratioDeaths65.push(curRatio65);
        ratioDeathsSum.push(curRatioSum);

        deathDatesRaw.push(curRow.date);
    }

    
    doneLoading = true;
    interpretDate();
    updateChart();
    
});

// Function for converting date-string into date format
var interpretDate = function(){
    for (let d = 0; d < allDatesRaw.length; d++) {
        const curDateRaw = allDatesRaw[d];
        const curDate = new Date(curDateRaw);
        allDates.push(curDate)
    }
    for (let d = 0; d < allDatesDeathRaw.length; d++) {
        const curDateRaw = allDatesDeathRaw[d];
        const curDate = new Date(curDateRaw);
        allDatesDeath.push(curDate)
    }
    for (let d = 0; d < deathDatesRaw.length; d++) {
        const curDateRaw = deathDatesRaw[d];
        const curDate = new Date(curDateRaw);
        deathDates.push(curDate)
    }
}

// Function for interpreting the x-range to show
// There are 365 data-points, slider goes from 0 to 365
var numToDate = function(curNum){
    
    return allDates[curNum];
}


var chartConfig = {
    type: 'line',
    data: {
        labels: allDates,
        datasets: [ // No data, is added in functions below
        ]
    },
    options: {
        responsive: true,
        aspectRatio: 3,
        legend: {
            position: 'top',
            labels: {
            usePointStyle: true,
            }
        },
        scales: {
            xAxes: [{
                type: 'time',
                    time: {
                        unit: 'month',
                        stepSize: 1,
                        // unit: 'week',
                        // tooltipFormat:'MM/DD/YYYY', 
                        // tooltipFormat:'DD/MM - YYYY', 
                        tooltipFormat:'[Uge ]w[, ] DD[.] MMMM - YYYY', 
                        // min: new Date('1917-01-01'), 
                        // max: curMaxDate
                        displayFormats: {
                            week: '[Uge ]w[, ]YYYY', 
                            month: 'MMMM YYYY'
                        }
                },
                display: true,
                scaleLabel: {
                display: true
                },            
                // ticks: {
                //     // min: curMinDate,
                //     min: new Date('1917-01-01'),
                //     max: curMaxDate
                // }
            }],
            yAxes: [{
                display: true,
                scaleLabel: {
                    fontSize: 20,
                    display: true,
                    labelString: 'Antal'
                }
            }]
        }
    }
};

var chartDeathConfig = {
    type: 'line',
    data: {
        labels: allDatesDeath,
        datasets: [ 
            {
                    label: 'Alle aldersgrupper',
                    data: allDeath,
                    // data: casesSum,
                    borderColor: window.chartColors.black,
                    backgroundColor:  window.chartColors.lightgrey,
                    fill: true,
                    showLine: true,
                    lineTension: 0,
                    borderWidth: 1,
                    pointRadius: 2,
                    pointHoverRadius: 5,
                    id: 'sumDeath',
            }
        ]
        // labels: deathDates,
        // datasets: [ 
        //     {
        //             label: 'Alle aldersgrupper',
        //             data: deathsSum,
        //             // data: casesSum,
        //             borderColor: window.chartColors.black,
        //             backgroundColor:  window.chartColors.lightgrey,
        //             fill: true,
        //             showLine: true,
        //             lineTension: 0,
        //             borderWidth: 1,
        //             pointRadius: 2,
        //             pointHoverRadius: 5,
        //             id: 'sumDeath',
        //     }
        // ]
    },
    options: {
        responsive: true,
        aspectRatio: 3,
        legend: {
            position: 'top',
            labels: {
            usePointStyle: true,
            }
        },
        scales: {
            xAxes: [{
                type: 'time',
                    time: {
                        unit: 'month',
                        stepSize: 1,
                        // unit: 'week',
                        // tooltipFormat:'MM/DD/YYYY', 
                        // tooltipFormat:'DD/MM - YYYY', 
                        tooltipFormat:'[Uge ]w[, ] DD[.] MMMM - YYYY', 
                        // min: new Date('1917-01-01'), 
                        // max: curMaxDate
                        displayFormats: {
                            week: '[Uge ]w[, ]YYYY', 
                            month: 'MMMM YYYY'
                        }
                },
                display: true,
                scaleLabel: {
                display: true
                },            
                // ticks: {
                //     // min: curMinDate,
                //     min: new Date('1917-01-01'),
                //     max: curMaxDate
                // }
            }],
            yAxes: [{
                display: true,
                scaleLabel: {
                    fontSize: 20,
                    display: true,
                    labelString: 'Daglige dødsfald'
                }
            }]
        }
    }
};

var chartDeath2Config = {
    type: 'bar',
    data: {
        labels: deathDates,
        // labels: allDatesDeath,
        // datasets: [ 
        //     {
        //             label: 'Alle aldersgrupper',
        //             data: deathsSum,
        //             // data: allDeath,
        //             borderColor: window.chartColors.black,
        //             backgroundColor:  window.chartColors.lightgrey,
        //             fill: true,
        //             showLine: true,
        //             lineTension: 0,
        //             borderWidth: 1,
        //             pointRadius: 2,
        //             pointHoverRadius: 5,
        //             id: 'sumDeath',
        //     }
        // ]
    },
    options: {
        responsive: true,
        aspectRatio: 3,
        legend: {
            position: 'top',
            labels: {
            usePointStyle: true,
            }
        },
        scales: {
            xAxes: [{
                type: 'time',
                    time: {
                        unit: 'month',
                        stepSize: 1,
                        // unit: 'week',
                        // tooltipFormat:'MM/DD/YYYY', 
                        // tooltipFormat:'DD/MM - YYYY', 
                        tooltipFormat:'[Uge ]w[, ] DD[.] MMMM - YYYY', 
                        // min: new Date('1917-01-01'), 
                        // max: curMaxDate
                        displayFormats: {
                            week: '[Uge ]w[, ]YYYY', 
                            month: 'MMMM YYYY'
                        }
                },
                display: true,
                scaleLabel: {
                display: true
                },            
                // ticks: {
                //     // min: curMinDate,
                //     min: new Date('1917-01-01'),
                //     max: curMaxDate
                // }
            }],
            yAxes: [{
                display: true,
                scaleLabel: {
                    fontSize: 20,
                    display: true,
                    labelString: 'Månedlige dødsfald'
                }
            }]
        }
    }
};
    
let saveButton = document.getElementById('saveButton');
let saveButton2 = document.getElementById('saveButton2');
let saveButton3 = document.getElementById('saveButton3');
let xlim1_slider = document.getElementById('xlim1');
let xlim1Label = document.getElementById('xlim1Label');
let xlim2_slider = document.getElementById('xlim2');
let xlim2Label = document.getElementById('xlim2Label');
let Checkbox0 = document.getElementById('Checkbox0');
let Checkbox1 = document.getElementById('Checkbox1');
let Checkbox5 = document.getElementById('Checkbox5');
let Checkbox15 = document.getElementById('Checkbox15');
let Checkbox65 = document.getElementById('Checkbox65');
let CheckboxSum = document.getElementById('CheckboxSum');
let CheckboxDeaths0 = document.getElementById('CheckboxDeaths0');
let CheckboxDeaths1 = document.getElementById('CheckboxDeaths1');
let CheckboxDeaths5 = document.getElementById('CheckboxDeaths5');
let CheckboxDeaths15 = document.getElementById('CheckboxDeaths15');
let CheckboxDeaths65 = document.getElementById('CheckboxDeaths65');
let CheckboxDeathsSum = document.getElementById('CheckboxDeathsSum');
let show0 = Checkbox0.checked;
let show1 = Checkbox1.checked;
let show5 = Checkbox5.checked;
let show15 = Checkbox15.checked;
let show65 = Checkbox65.checked;
let showSum = CheckboxSum.checked;
let showDeaths0 = CheckboxDeaths0.checked;
let showDeaths5 = CheckboxDeaths5.checked;
let showDeaths15 = CheckboxDeaths15.checked;
let showDeaths65 = CheckboxDeaths65.checked;
let showDeathsSum = CheckboxDeathsSum.checked;

let radioNorm = document.getElementsByName('norm');

// function addData(chart, label, data) {
//     chart.data.labels.push(label);
//     chart.data.datasets.forEach((dataset) => {
//         dataset.data.push(data);
//     });
//     chart.update();
// }

// function removeData(chart){
//     chart.data.labels.pop();
//     chart.data.datasets.forEach((dataset) => {
//         dataset.data.pop();
//     });
//     chart.update();
// }

// On loading the page, make the chartjs figures
window.onload = function() {
    var ctx = document.getElementById('canvas').getContext('2d');
    window.mainChart = new Chart(ctx, chartConfig);
    var ctxDeath = document.getElementById('canvasDeath').getContext('2d');
    window.DeathChart = new Chart(ctxDeath, chartDeathConfig);
    var ctxDeath2 = document.getElementById('canvasDeath2').getContext('2d');
    window.DeathChart2 = new Chart(ctxDeath2, chartDeath2Config);
    updateChart();

    // // Define the data
    // var sumDeath = {
    //     label: 'Alle aldersgrupper',
    //     data: allDeath,
    //     // data: casesSum,
    //     borderColor: window.chartColors.black,
    //     backgroundColor:  window.chartColors.black,
    //     fill: false,
    //     showLine: true,
    //     lineTension: 0,
    //     pointRadius: 2,
    //     pointHoverRadius: 5,
    //     id: 'sumDeath',
    // };
    // // Add to chart
    // chartDeathConfig.data.datasets.push(sumDeath);

    // Initialize according to defaults set in html document
    Checkbox0.onchange();
    Checkbox1.onchange();
    Checkbox5.onchange();
    Checkbox15.onchange();
    Checkbox65.onchange();
    CheckboxSum.onchange();
    CheckboxDeaths0.onchange();
    CheckboxDeaths5.onchange();
    CheckboxDeaths15.onchange();
    CheckboxDeaths65.onchange();
    CheckboxDeathsSum.onchange();
    // xlim1_slider.onchange();
    // xlim2_slider.onchange();
    
}

var updateChart = function(){

    curMinDate = numToDate(xlim1_slider.value);
    curMaxDate = numToDate(xlim2_slider.value);
    
    window.mainChart.options.scales.xAxes[0].time.min = curMinDate;
    window.mainChart.options.scales.xAxes[0].time.max = curMaxDate;
    window.DeathChart.options.scales.xAxes[0].time.min = curMinDate;
    window.DeathChart.options.scales.xAxes[0].time.max = curMaxDate;
    window.DeathChart2.options.scales.xAxes[0].time.min = curMinDate;
    window.DeathChart2.options.scales.xAxes[0].time.max = curMaxDate;

    if (showRatioK){
        window.mainChart.options.scales.yAxes[0].scaleLabel.labelString = 'Tilfælde per 1000';
        window.DeathChart2.options.scales.yAxes[0].scaleLabel.labelString = 'Månedlige dødsfald per 1000';
    } else {
        window.mainChart.options.scales.yAxes[0].scaleLabel.labelString = 'Antal tilfælde';
        window.DeathChart2.options.scales.yAxes[0].scaleLabel.labelString = 'Månedlige dødsfald';
    }

    // Update chart
    window.mainChart.update();
    window.DeathChart.update();
    window.DeathChart2.update();
}

// function addAllDeathData(){
//     // Define the data
//     var sumDeath = {
//         label: 'Alle aldersgrupper',
//         data: allDeath,
//         // data: casesSum,
//         borderColor: window.chartColors.black,
//         backgroundColor:  window.chartColors.black,
//         fill: false,
//         showLine: true,
//         lineTension: 0,
//         pointRadius: 2,
//         pointHoverRadius: 5,
//         id: 'sumDeath',
//     };
//     // Add to chart
//     chartDeathConfig.data.datasets.push(sumDeath);

// }

// currentValue = 0
function handleClick(myRadio) {
    if (myRadio.value == 'ratioK'){
        showRatioK = true;
    } else {
        showRatioK = false;
    }
    
    // Clear all datasets
    for (let k = chartConfig.data.datasets.length ; k > 0; k--) {
        chartConfig.data.datasets.pop()   
    }
    
    for (let k = chartDeath2Config.data.datasets.length ; k > 0; k--) {
        chartDeath2Config.data.datasets.pop()   
    }

    // Add them again
    Checkbox0.onchange();
    Checkbox1.onchange();
    Checkbox5.onchange();
    Checkbox15.onchange();
    Checkbox65.onchange();
    CheckboxSum.onchange();
    Checkbox1.onchange();
    CheckboxDeaths0.onchange();
    CheckboxDeaths5.onchange();
    CheckboxDeaths15.onchange();
    CheckboxDeaths65.onchange();
    CheckboxDeathsSum.onchange();
    
    
    updateChart();
}
function fillCanvasBackgroundWithColor(canvas, color) {
    // From https://stackoverflow.com/questions/50104437/set-background-color-to-save-canvas-chart/50126796#50126796

    // Get the 2D drawing context from the provided canvas.
    const context = canvas.getContext('2d');
  
    // We're going to modify the context state, so it's
    // good practice to save the current state first.
    context.save();
  
    // Normally when you draw on a canvas, the new drawing
    // covers up any previous drawing it overlaps. This is
    // because the default `globalCompositeOperation` is
    // 'source-over'. By changing this to 'destination-over',
    // our new drawing goes behind the existing drawing. This
    // is desirable so we can fill the background, while leaving
    // the chart and any other existing drawing intact.
    // Learn more about `globalCompositeOperation` here:
    // https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/globalCompositeOperation
    context.globalCompositeOperation = 'destination-over';
  
    // Fill in the background. We do this by drawing a rectangle
    // filling the entire canvas, using the provided color.
    context.fillStyle = color;
    context.fillRect(0, 0, canvas.width, canvas.height);
  
    // Restore the original context state from `context.save()`
    context.restore();
  }

saveButton.onclick = function(){
    // Hacked together way for saving the figure using just javascript

    // Set the background color 
    // (From https://stackoverflow.com/questions/50104437/set-background-color-to-save-canvas-chart/50126796#50126796 )
    const canvas = document.getElementById('canvas');
    fillCanvasBackgroundWithColor(canvas, 'white');

    // Convert to image
    var url  = window.mainChart.toBase64Image();

    // Make a link element for download, click it, delete it again
    // (From second answer of https://stackoverflow.com/questions/17311645/download-image-with-javascript )
    var a = document.createElement('a');
    a.href = url;
    a.download = "SpanskSygeTilfaelde.png";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    
    
}
// Save the second figure as well
saveButton2.onclick = function(){
    const canvasDeath2 = document.getElementById('canvasDeath2');
    fillCanvasBackgroundWithColor(canvasDeath2, 'white');
    var url = window.DeathChart2.toBase64Image();
    var  a = document.createElement('a');
    a.href = url;
    a.download = "SpanskSygeDoedsfald.png";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}
// Save the second figure as well
saveButton3.onclick = function(){
    const canvasDeath = document.getElementById('canvasDeath');
    fillCanvasBackgroundWithColor(canvasDeath, 'white');
    var url = window.DeathChart.toBase64Image();
    var  a = document.createElement('a');
    a.href = url;
    a.download = "SpanskSygeDoedsfaldDagligt.png";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

xlim1_slider.onchange = function(){
    curMinDate = numToDate(xlim1_slider.value);

    // xlim1Label.innerHTML ='Første dag: '+curMinDate.getDate()+'/'+curMinDate.getMonth()+' - 19'+curMinDate.getYear();
    var minDateMoment = moment(curMinDate)
    xlim1Label.innerHTML ='Første dag: '+minDateMoment.format('DD[. ]MMM[ - ]YYYY');
    
    // Set the minimum of xlim2 slider
    var minOnMax = parseInt(xlim1_slider.value)
    xlim2_slider.setAttribute("min", minOnMax+1);

    // Update chart
    updateChart();

}
xlim2_slider.onchange = function(){
    curMaxDate = numToDate(xlim2_slider.value);

    // xlim2Label.innerHTML ='Sidste dag: '+curMaxDate.getDate()+'/'+curMaxDate.getMonth()+' - 19'+curMaxDate.getYear();
    var maxDateMoment = moment(curMaxDate)
    xlim2Label.innerHTML ='Sidste dag: '+maxDateMoment.format('DD[. ]MMM[ - ]YYYY');
    
    // Set the maximum of xlim2 slider
    var maxOnMin = parseInt(xlim2_slider.value)
    xlim1_slider.setAttribute("max", maxOnMin-1);

    // Update chart
    updateChart();

}



CheckboxSum.onchange = function(){
    showSum = CheckboxSum.checked;

    var curData;
    if (showRatioK){
        curData = ratioKSum
    } else {
        curData = casesSum
    }

    // Summed data
    if (showSum){
        // Define the data
        var sumData =     {
            label: 'Alle aldersgrupper',
            data: curData,
            // data: casesSum,
            borderColor: window.chartColors.black,
            backgroundColor:  window.chartColors.black,
            fill: false,
            showLine: true,
            lineTension: 0,
            pointRadius: 2,
            pointHoverRadius: 5,
            id: 'sumData',
        };
        // Add to chart
        chartConfig.data.datasets.push(sumData);

    } else {
        // Look for the dataset
        chartConfig.data.datasets.find((dataset, index) => {
            // When found, delete it
            if (dataset.id === 'sumData') {
                chartConfig.data.datasets.splice(index, 1);
                return true; // stop searching
            }
        });
    }

    // Update the chart
    updateChart();
}
Checkbox0.onchange = function(){
    show0 = Checkbox0.checked;
    
    var curData;
    if (showRatioK){
        curData = ratioK0_1
    } else {
        curData = cases0_1
    }

    // 0 - 1 årige
    if (show0){
        // Define the data
        var data0 =     {
            label: '< 1 år',
            data: curData,
            borderColor: window.chartColors.blue,
            backgroundColor:  window.chartColors.blue,
            fill: false,
            showLine: true,
            lineTension: 0,
            pointRadius: 2,
            pointHoverRadius: 5,
            id: 'data0',
        };
        // Add to chart
        chartConfig.data.datasets.push(data0);

    } else {
        // Look for the dataset
        chartConfig.data.datasets.find((dataset, index) => {
            // When found, delete it
            if (dataset.id === 'data0') {
                chartConfig.data.datasets.splice(index, 1);
                return true; // stop searching
            }
        });
    }
    updateChart();
}
Checkbox1.onchange = function(){
    show1 = Checkbox1.checked;
        
    var curData;
    if (showRatioK){
        curData = ratioK1_4
    } else {
        curData = cases1_4
    }

    // 1 - 4 årige
    if (show1){
        // Define the data
        var data1 =     {
            label: '1 - 4 år',
            data: curData,
            borderColor: window.chartColors.green,
            backgroundColor:  window.chartColors.green,
            fill: false,
            showLine: true,
            lineTension: 0,
            pointRadius: 2,
            pointHoverRadius: 5,
            id: 'data1',
        };
        // Add to chart
        chartConfig.data.datasets.push(data1);

    } else {
        // Look for the dataset
        chartConfig.data.datasets.find((dataset, index) => {
            // When found, delete it
            if (dataset.id === 'data1') {
                chartConfig.data.datasets.splice(index, 1);
                return true; // stop searching
            }
        });
    }

    updateChart();
}
Checkbox5.onchange = function(){
    show5 = Checkbox5.checked;
        
    var curData;
    if (showRatioK){
        curData = ratioK5_14
    } else {
        curData = cases5_14
    }

    // 5 - 14 årige
    if (show5){
        // Define the data
        var data5 =     {
            label: '5 - 14 år',
            data: curData,
            borderColor: window.chartColors.yellow,
            backgroundColor:  window.chartColors.yellow,
            fill: false,
            showLine: true,
            lineTension: 0,
            pointRadius: 2,
            pointHoverRadius: 5,
            id: 'data5',
        };
        // Add to chart
        chartConfig.data.datasets.push(data5);
        
        // // Define the data
        // var data5death =     {
        //     label: '5 - 14 år',
        //     data: deaths5_14,
        //     borderColor: window.chartColors.yellow,
        //     backgroundColor:  window.chartColors.yellow,
        //     fill: false,
        //     showLine: true,
        //     lineTension: 0,
        //     pointRadius: 2,
        //     pointHoverRadius: 5,
        //     id: 'date5death',
        // };
        // // Add to chart
        // chartDeathConfig.data.datasets.push(data5death);

    } else {
        // Look for the dataset
        chartConfig.data.datasets.find((dataset, index) => {
            // When found, delete it
            if (dataset.id === 'data5') {
                chartConfig.data.datasets.splice(index, 1);
                return true; // stop searching
            }
        });
        // // Look for the dataset
        // chartDeathConfig.data.datasets.find((dataset, index) => {
        //     // When found, delete it
        //     if (dataset.id === 'date5death') {
        //         chartDeathConfig.data.datasets.splice(index, 1);
        //         return true; // stop searching
        //     }
        // });
    }
    
    updateChart();
}
Checkbox15.onchange = function(){
    show15 = Checkbox15.checked;
        
    var curData;
    if (showRatioK){
        curData = ratioK15_64
    } else {
        curData = cases15_64
    }

    // 15 - 64 årige
    if (show15){
        // Define the data
        var data15 =     {
            label: '15 - 64 år',
            data: curData,
            borderColor: window.chartColors.purple,
            backgroundColor:  window.chartColors.purple,
            fill: false,
            showLine: true,
            lineTension: 0,
            pointRadius: 2,
            pointHoverRadius: 5,
            id: 'data15',
        };
        // Add to chart
        chartConfig.data.datasets.push(data15);
        
        // // Define the data
        // var data15death =     {
        //     label: '15 - 64 år',
        //     data: deaths15_64,
        //     borderColor: window.chartColors.purple,
        //     backgroundColor:  window.chartColors.purple,
        //     fill: false,
        //     showLine: true,
        //     lineTension: 0,
        //     pointRadius: 2,
        //     pointHoverRadius: 5,
        //     id: 'data15death',
        // };
        // // Add to chart
        // chartDeathConfig.data.datasets.push(data15death);

    } else {
        // Look for the dataset
        chartConfig.data.datasets.find((dataset, index) => {
            // When found, delete it
            if (dataset.id === 'data15') {
                chartConfig.data.datasets.splice(index, 1);
                return true; // stop searching
            }
        });
        // // Look for the dataset
        // chartDeathConfig.data.datasets.find((dataset, index) => {
        //     // When found, delete it
        //     if (dataset.id === 'data15death') {
        //         chartDeathConfig.data.datasets.splice(index, 1);
        //         return true; // stop searching
        //     }
        // });
    }
    
    updateChart();
}
Checkbox65.onchange = function(){
    show65 = Checkbox65.checked;
        
    var curData;
    if (showRatioK){
        curData = ratioK65
    } else {
        curData = cases65
    }

    // 65+ årige
    if (show65){
        // Define the data
        var data65 =     {
            label: '65+ år',
            data: curData,
            borderColor: window.chartColors.red,
            backgroundColor:  window.chartColors.red,
            fill: false,
            showLine: true,
            lineTension: 0,
            pointRadius: 2,
            pointHoverRadius: 5,
            id: 'data65',
        };
        // Add to chart
        chartConfig.data.datasets.push(data65);
    } else {
        // Look for the dataset
        chartConfig.data.datasets.find((dataset, index) => {
            // When found, delete it
            if (dataset.id === 'data65') {
                chartConfig.data.datasets.splice(index, 1);
                return true; // stop searching
            }
        });
    }
    
    updateChart();
}

CheckboxDeathsSum.onchange = function(){
    showDeathsSum = CheckboxDeathsSum.checked;

    var curData;
    if (showRatioK){
        curData = ratioDeathsSum
    } else {
        curData = deathsSum
    }

    // Summed data
    if (showDeathsSum){
        // Define the data
        var sumData =     {
            label: 'Alle aldersgrupper',
            data: curData,
            borderColor: window.chartColors.black,
            backgroundColor:  window.chartColors.black,
            fill: false,
            showLine: true,
            lineTension: 0,
            pointRadius: 2,
            pointHoverRadius: 5,
            id: 'sumDataDeaths',
        };
        // Add to chart
        chartDeath2Config.data.datasets.push(sumData);

    } else {
        // Look for the dataset
        chartDeath2Config.data.datasets.find((dataset, index) => {
            // When found, delete it
            if (dataset.id === 'sumDataDeaths') {
                chartDeath2Config.data.datasets.splice(index, 1);
                return true; // stop searching
            }
        });
    }

    // Update the chart
    updateChart();
}

CheckboxDeaths65.onchange = function(){
    showDeaths65 = CheckboxDeaths65.checked;

    var curData;
    if (showRatioK){
        curData = ratioDeaths65
    } else {
        curData = deaths65
    }

    // Summed data
    if (showDeaths65){
        // Define the data
        var sumData =     {
            label: '65+ årige',
            data: curData,
            borderColor: window.chartColors.red,
            backgroundColor:  window.chartColors.red,
            fill: false,
            showLine: true,
            lineTension: 0,
            pointRadius: 2,
            pointHoverRadius: 5,
            id: 'DataDeaths65',
        };
        // Add to chart
        chartDeath2Config.data.datasets.push(sumData);

    } else {
        // Look for the dataset
        chartDeath2Config.data.datasets.find((dataset, index) => {
            // When found, delete it
            if (dataset.id === 'DataDeaths65') {
                chartDeath2Config.data.datasets.splice(index, 1);
                return true; // stop searching
            }
        });
    }

    // Update the chart
    updateChart();
}

CheckboxDeaths15.onchange = function(){
    showDeaths15 = CheckboxDeaths15.checked;

    var curData;
    if (showRatioK){
        curData = ratioDeaths15_64
    } else {
        curData = deaths15_64
    }

    // Summed data
    if (showDeaths15){
        // Define the data
        var sumData =     {
            label: '15 til 64 årige',
            data: curData,
            borderColor: window.chartColors.purple,
            backgroundColor:  window.chartColors.purple,
            fill: false,
            showLine: true,
            lineTension: 0,
            pointRadius: 2,
            pointHoverRadius: 5,
            id: 'DataDeaths15',
        };
        // Add to chart
        chartDeath2Config.data.datasets.push(sumData);

    } else {
        // Look for the dataset
        chartDeath2Config.data.datasets.find((dataset, index) => {
            // When found, delete it
            if (dataset.id === 'DataDeaths15') {
                chartDeath2Config.data.datasets.splice(index, 1);
                return true; // stop searching
            }
        });
    }

    // Update the chart
    updateChart();
}

CheckboxDeaths5.onchange = function(){
    showDeaths5 = CheckboxDeaths5.checked;

    var curData;
    if (showRatioK){
        curData = ratioDeaths5_14
    } else {
        curData = deaths5_14
    }

    // Summed data
    if (showDeaths5){
        // Define the data
        var sumData =     {
            label: '5 til 14 årige',
            data: curData,
            borderColor: window.chartColors.yellow,
            backgroundColor:  window.chartColors.yellow,
            fill: false,
            showLine: true,
            lineTension: 0,
            pointRadius: 2,
            pointHoverRadius: 5,
            id: 'DataDeaths5',
        };
        // Add to chart
        chartDeath2Config.data.datasets.push(sumData);

    } else {
        // Look for the dataset
        chartDeath2Config.data.datasets.find((dataset, index) => {
            // When found, delete it
            if (dataset.id === 'DataDeaths5') {
                chartDeath2Config.data.datasets.splice(index, 1);
                return true; // stop searching
            }
        });
    }

    // Update the chart
    updateChart();
}

CheckboxDeaths0.onchange = function(){
    showDeaths0 = CheckboxDeaths0.checked;

    var curData;
    if (showRatioK){
        curData = ratioDeaths0_4
    } else {
        curData = deaths0_4
    }

    // Summed data
    if (showDeaths0){
        // Define the data
        var sumData =     {
            label: '0 til 4 årige',
            data: curData,
            borderColor: window.chartColors.green,
            backgroundColor:  window.chartColors.green,
            fill: false,
            showLine: true,
            lineTension: 0,
            pointRadius: 2,
            pointHoverRadius: 5,
            id: 'DataDeaths0',
        };
        // Add to chart
        chartDeath2Config.data.datasets.push(sumData);

    } else {
        // Look for the dataset
        chartDeath2Config.data.datasets.find((dataset, index) => {
            // When found, delete it
            if (dataset.id === 'DataDeaths0') {
                chartDeath2Config.data.datasets.splice(index, 1);
                return true; // stop searching
            }
        });
    }

    // Update the chart
    updateChart();
}


// , function(d) {
//     return {
//         date: d.casesSum,
//     //   year: new Date(+d.Year, 0, 1), // convert "Year" column to Date
//     //   make: d.Make,
//     //   model: d.Model,
//     //   length: +d.Length // convert "Length" column to number
//     };
//   }, function(error, rows) {
//     console.log(rows);
//   });
// d3.csv("Spansk_Syge_I_Cph.csv").row(function(d) { return {key: d.key, value: +d.value}; }).get(function(error, rows) { console.log(rows); });
//  // Plot the data with Chart.js
//  function makeChart(countries) {
//    var countryLabels = date.map(function (d) {
//      return d.date;
//    });
//    var populationData = casesSum.map(function (d) {
//      return d.casesSum;
//    });

//    var chart = new Chart("myChart", {
//      type: "bar",
//      data: {
//        labels: countryLabels,
//        datasets: [
//          {
//            data: populationData 
//          }
//        ]
//      }
//    });
//  }