// Copyright (C) 2021 Yu Yang
//
// This file is part of Vesyla.
//
// Vesyla is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Vesyla is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with Vesyla.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __VESYLA_SCHEDULE_VE_CONFIG_HPP__
#define __VESYLA_SCHEDULE_VE_CONFIG_HPP__

#define VERSION_MAJOR 2
#define VERSION_MINOR 2
#define VERSION_PATCH 0

const char *LICENSE_NAME = R"(GPL v3)";
const char *LICENSE_DESC = R"(
Copyright (C) 2017-2021  Yu Yang <yuyang2@kth.se>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
)";

const char *LOGGING_CONF = R"(
* GLOBAL:
   FORMAT               =  "[%level %datetime{%H:%m:%s}] %msg"
   FILENAME             =  "/tmp/vesyla-suite/vs-init.log"
   ENABLED              =  true
   TO_FILE              =  true
   TO_STANDARD_OUTPUT   =  true
   SUBSECOND_PRECISION  =  3
   PERFORMANCE_TRACKING =  false
   MAX_LOG_FILE_SIZE    =  2097152
   LOG_FLUSH_THRESHOLD  =  100
* DEBUG:
   FORMAT               =  "[%level %datetime{%H:%m:%s}] (%func @ %loc) %msg"
)";

#endif // __VESYLA_SCHEDULE_VE_CONFIG_HPP__
