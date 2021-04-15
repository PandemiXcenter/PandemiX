
Highcharts.chart('containerNumTests', {

    title: {
        text: 'COVID-19'
    },

    // subtitle: {
    //     text: 'Antigen tests'
    // },

    data: {
        // csvURL: window.location.origin + '/web/NumTests.csv'
        csvURL: 'web/NumTests.csv'
    },

    plotOptions: {
        series: {
            marker: {
                enabled: false
            }
        }
    },

    xAxis: {
        type: 'datetime'
    },
    yAxis: {
        title: {
            text: 'Antal af test'
        },
    }
});



Highcharts.chart('containerPosPct', {

    title: {
        text: 'COVID-19'
    },

    // subtitle: {
    //     text: 'Antigen tests'
    // },

    data: {
        csvURL: 'web/PosPct.csv'
        // csvURL: window.location.origin + '/web/PosPct.csv'
        // csv: '/web/AntigenTestsCleaned.csv'
        // csv: document.getElementById('csv').innerHTML
    },

    plotOptions: {
        series: {
            marker: {
                enabled: false
            }
        }
    },

    xAxis: {
        type: 'datetime'
    },

    yAxis: {
        title: {
            text: 'Andel af test [%]'
        },
    }
});
Highcharts.chart('containerConfirmed', {

    title: {
        text: 'COVID-19'
    },

    // subtitle: {
    //     text: 'Antigen tests'
    // },

    data: {
        csvURL: 'web/PCRconfirmed.csv'
        // csvURL: window.location.origin + '/web/PCRconfirmed.csv'
    },

    plotOptions: {
        series: {
            marker: {
                enabled: false
            }
        }
    },

    xAxis: {
        type: 'datetime'
    },
    yAxis: {
        title: {
            text: 'Andel af PCR-konfirmerede test [%]'
        },
    }
});
